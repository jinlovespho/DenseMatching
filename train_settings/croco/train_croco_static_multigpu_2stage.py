from termcolor import colored
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image



from utils_data.image_transforms import ArrayToTensor
from training.actors.batch_processing import GLUNetBatchPreprocessing
from training.losses.basic_losses import EPE
from training.losses.multiscale_loss import MultiScaleFlow, MultiScaleMixtureDensity
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from training.losses.neg_log_likelihood import NLLMixtureLaplace
from admin.multigpu import MultiGPU
from training.actors.self_supervised_actor import CrocoBasedActor
from datasets.load_pre_made_datasets.load_pre_made_dataset import PreMadeDataset
from datasets.object_augmented_dataset import MSCOCO, AugmentedImagePairsDatasetMultipleObjects
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import RandomAffine
from datasets.MegaDepth.megadepth import MegaDepthDataset
from datasets.mixture_of_datasets import MixDatasets
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.pixel_wise_mapping import warp


# JLP
import os
import torch 
import wandb
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import datetime
import torchvision

def load_network(net, checkpoint_path=None, **kwargs):
    """Loads a network checkpoint file.
    args:
        net: network architecture
        checkpoint_path: ~~~.pth.tar
    outputs:
        net: loaded network
    """
    if not os.path.isfile(checkpoint_path): 
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(checkpoint_path))

    # Load checkpoint
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']

    msg=net.load_state_dict(checkpoint_dict, strict=False)
    # print(msg)
    print('---------------------------------------')
    print('Weight Loaded from .tar !')
    print('Checkpoint Path: ', checkpoint_path)
    print('missing keys: ', msg.missing_keys) # model 에는 있는데 ckpt 에는 없는 것들
    print('unexpected keys: ', msg.unexpected_keys) # ckpt 에는 있는데 model 에는 없는 것들
    print('---------------------------------------')
    return net

def run(settings, args):
    settings.description = 'Train setting for croco on dataset static dataset called DPED-CityScape-ADE'
    settings.data_mode = 'local'
    settings.batch_size = args.batch_size
    settings.n_threads = 8
    settings.multi_gpu = args.multi_gpu
    settings.lr = args.lr
    settings.scheduler_steps = [65, 75, 95]
    settings.n_epochs = args.max_epoch

    if settings.multi_gpu:
        args.rank = int(os.environ["RANK"])                 # global rank
        args.world_size = int(os.environ['WORLD_SIZE'])     # num gpus
        args.gpu = int(os.environ['LOCAL_RANK'])            # local rank
        args.dist_url = 'env://'
        args.dist_backend = 'nccl'
        
        torch.cuda.set_device(args.gpu)
        
        print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.rank) #, timeout=datetime.timedelta(minutes=10))
        assert dist.is_available() and dist.is_initialized(), 'Distributed training has not been initialized'
        print('distributed training has been WELL initialized')
    
    # 0. Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    settings.nbr_objects = 4
    settings.min_area_objects = 1300
    # perturbations
    perturbations_parameters_v2 = {'elastic_param': {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1,
                                                     "max_alpha": 0.4},
                                   'max_sigma_mask': 10, 'min_sigma_mask': 3}
    # very important, we compute the object reprojection mask, will be used for training
    settings.compute_object_reprojection_mask = True

    # 1. Define training and validation datasets
    # Train dataset: combination of synthetic data with perturbations + independently moving objects, and real image
    # pairs from the MegaDepth dataset with sparse ground-truth matches

    # 1st training dataset: synthetic data with perturbations + independently moving objects
    # object foreground dataset
    fg_tform = RandomAffine(p_flip=0.0, max_rotation=30.0,
                            max_shear=0, max_ar_factor=0.,
                            max_scale=0.3, pad_amount=0)

    coco_dataset_train = MSCOCO(root=settings.env.coco, split='train', version='2014',
                                min_area=settings.min_area_objects)

    # base dataset with image pairs and ground-truth flow field + adding perturbations
    train_dataset_, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                       source_image_transform=None,
                                       target_image_transform=None,
                                       flow_transform=None,
                                       co_transform=None,
                                       split=1,
                                       get_mapping=False,
                                       add_discontinuity=True,
                                       parameters_v2=perturbations_parameters_v2,
                                       max_nbr_perturbations=15,
                                       min_nbr_perturbations=5)  # only training

    # add independently moving objects + compute the reprojection mask
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    co_transform = None
    train_dataset_dynamic = AugmentedImagePairsDatasetMultipleObjects(
        foreground_image_dataset=coco_dataset_train, background_image_dataset=train_dataset_,
        foreground_transform=fg_tform, source_image_transform=source_img_transforms,
        target_image_transform=target_img_transforms, flow_transform=flow_transform,
        co_transform=co_transform, number_of_objects=settings.nbr_objects, image_size=(224,224),
        compute_object_reprojection_mask=settings.compute_object_reprojection_mask)

    # 2nd training dataset: MegaDepth data
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])

    megadepth_cfg = {'scene_info_path': os.path.join(settings.env.megadepth_training, 'scene_info'),
                     'train_num_per_scene': 300, 'val_num_per_scene': 25,
                     'output_image_size': [224, 224], 'pad_to_same_shape': True,
                     'output_flow_size': [[224, 224], [224, 224]]}
    training_dataset_megadepth = MegaDepthDataset(root=settings.env.megadepth_training, cfg=megadepth_cfg,
                                                  source_image_transform=source_img_transforms,
                                                  target_image_transform=source_img_transforms,
                                                  flow_transform=flow_transform, co_transform=co_transform,
                                                  split='train', store_scene_info_in_memory=False)
    
    # data = training_dataset_megadepth[1]
    # data = training_dataset_megadepth[0]
    # torchvision.utils.save_image(data['source_image']/255., 'source_image.png')
    # torchvision.utils.save_image(data['target_image']/255., 'target_image.png')
    # torchvision.utils.save_image(data['occlusion_mask'][0].float(), 'occlusion_mask.png')
    # torchvision.utils.save_image(data['occlusion_mask'][0]*(data['source_image']/255.), 'occlusion_masked_source_image.png')
    # torchvision.utils.save_image(data['occlusion_mask'][0]*(data['target_image']/255.), 'occlusion_masked_target_image.png')
    # flow_vis = flow_to_image(data['flow_map'][0].permute(1,2,0).numpy())
    # flow_fw = Image.fromarray(flow_vis)
    # flow_fw.save('./flow_fw.png')
    # warped_target = warp(data['source_image'].unsqueeze(dim=0)/255., data['flow_map'][0].unsqueeze(dim=0))
    # torchvision.utils.save_image(warped_target, 'warped_target.png')
    # torchvision.utils.save_image(warped_target * data['occlusion_mask'][0], 'masked_warped_target.png')
        
    # put store_scene_info_in_memory to True if more than 55GB of cpu memory is available. Sampling will be faster
    # final training dataset: combination of both previous datasets
    train_dataset = MixDatasets(list_of_datasets=[train_dataset_dynamic, training_dataset_megadepth],
                                list_overwrite_mask=[False, False], list_sparse=[False, True])
    

    # validation data
    megadepth_cfg['exchange_images_with_proba'] = 0.
    val_dataset = MegaDepthDataset(root=settings.env.megadepth_training,
                                   cfg=megadepth_cfg, split='val',
                                   source_image_transform=source_img_transforms,
                                   target_image_transform=source_img_transforms,
                                   flow_transform=flow_transform, co_transform=co_transform,
                                   store_scene_info_in_memory=False)
    
    if settings.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size, shuffle=False,
                          drop_last=False, training=True, num_workers=settings.n_threads, sampler=train_sampler)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads, sampler=val_sampler)

    # 3. Define model
    if args.model == 'crocoflow':
        from models.orig_croco.models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
        from models.orig_croco.models.head_downstream import PixelwiseTaskWithDPT
        ckpt = torch.load(args.croco_ckpt, 'cpu')   # crocoflow.pth
        ckpt_args = ckpt['args']
        ckpt_args.croco_args['img_size'] = args.img_size if args.img_size is not None else [320, 384]
        task = ckpt_args.task   # 'flow'
        tile_conf_mode = ckpt_args.tile_conf_mode   # tile_conf_mode='conf_expsigmoid_10_5'
        num_channels = {'stereo': 1, 'flow': 2}[task]   # 2
        with_conf = True
        if with_conf: num_channels += 1
        if dist.get_rank() == 0:
            print('head: PixelwiseTaskWithDPT()')
        head = PixelwiseTaskWithDPT()
        head.num_channels = num_channels
        if dist.get_rank() == 0:
            print('croco_args:', ckpt_args.croco_args)
        croco_args = ckpt_args.croco_args
        model = CroCoDownstreamBinocular(head, **croco_args)
        msg = model.load_state_dict(ckpt['model'], strict=True)
        if dist.get_rank() == 0:
            print('CROCO WEIGHT WELL LOADED: ', msg)
        model = model.to(device)
        model.train()
              
    elif args.model =='croco_catseg':
        from models.croco.croco import CroCoNet
        from models.croco.croco_downstream import croco_args_from_ckpt

        ckpt = torch.load(args.croco_ckpt, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = args.img_size
        croco_args['args'] = args
        model = CroCoNet(**croco_args)
        msg=model.load_state_dict(ckpt['model'], strict=False)
        if dist.get_rank() == 0:
            print('CROCO WEIGHT WELL LOADED: ', msg)
        model = model.to(device)
        model.train()
        
        if settings.pretrained_path is not None:
            print('Loading pretrained model from: ', settings.pretrained_path)
            pretrained_ckpt = torch.load(settings.pretrained_path, 'cpu')
            model.load_state_dict(pretrained_ckpt['state_dict'], strict=False)
            print('Pretrained model loaded!')
        
    elif args.model == 'crocov2':
        from models.croco.croco import CroCoNet
        from models.croco.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular

        # weights_already_loaded = True
        estimate_uncertainty = args.uncertainty

        # breakpoint()
        ckpt = torch.load(args.croco_ckpt,'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = ((args.img_size[0]//32)*32,(args.img_size[1]//32)*32)
        print('TRAINING IMG SIZE: ', croco_args['img_size'])
        croco_args['args'] = args
        model = CroCoNet(**croco_args)
        msg=model.load_state_dict(ckpt['model'], strict=False)
        print('missing keys: ', msg.missing_keys) # model 에는 있는데 ckpt 에는 없는 것들
        print('unexpected keys: ', msg.unexpected_keys) # ckpt 에는 있는데 model 에는 없는 것들
        print('CROCOV2 WEIGHT WELL LOADED: ', msg)
        model.train()
        model = model.to(device)

    else:
        raise NotImplementedError(f'Model {args.model} not selected!')

    path_to_pre_trained_models = args.path_to_pre_trained_models
    if path_to_pre_trained_models is not None:
        if path_to_pre_trained_models.endswith('.pth') or path_to_pre_trained_models.endswith('.pth.tar') or path_to_pre_trained_models.endswith('.pt'):    # true
            # if the path already corresponds to a checkpoint path, we use it directly
            checkpoint_fname = path_to_pre_trained_models
        model = load_network(model, checkpoint_path=checkpoint_fname)
        model.train()
        model = model.to(device)
    else:
        print('----------------------------------------------------------------') 
        print('NO PATH TO PRE-TRAINED MODELS!!!')
        print('----------------------------------------------------------------') 

    print('----------------------------------------------------------------') 

    # 3-2. Set Trainable Parameters
    if args.freeze == 'croco_enc':
        print('Freezing encoder parameters!')
        # Freeze parameters
        for name, param in model.named_parameters():
            if 'enc_blocks' in name or 'enc_norm' in name:
                param.requires_grad = False

    elif args.freeze == 'croco_all':
        print('Freezing all croco parameters!')
        for name, param in model.named_parameters():
            # for croco_catseg
            if 'cats_swin_decoder' in name: 
                param.requires_grad = True
            # for crocoflow
            elif 'head' in name:    
                param.requires_grad = True

            else:
                param.requires_grad = False
    else:
        print('Full Fine Tuning!')
    
    if args.lora_dec:
        print('Lora for only decoder!')
    
    # 3-3. Show params and trainable params
    tot_params = sum(p.numel() for p in model.parameters()) 
    tot_model_size = sum(p.numel()*p.element_size() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_model_size = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    print('----------------------------------------------------------------')   
    print(f"TOTAL PARAMS: {tot_params/1e6:.2f} M, TOTAL MODEL SIZE: {tot_model_size/1e6:.2f} MB")
    print(f"TRAINABLE PARAMS: {trainable_params/1e6:.2f} M, TRAINABLE MODEL SIZE: {trainable_model_size/1e6:.2f} MB")
    print('----------------------------------------------------------------')

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        # model = MultiGPU(model)        

    # 4. Define batch_processing
    batch_processing = GLUNetBatchPreprocessing(settings, apply_mask=args.apply_coco_msk, apply_mask_zero_borders=False,
                                                sparse_ground_truth=True, mapping=False)

    if args.model == 'crocoflow':
        weights_level_loss = [0.32]
    elif args.model == 'croco_catseg':
        weights_level_loss = [0.32, 0.32]
    elif args.model == 'crocov2':
        weights_level_loss = [0.32]
    
    if args.uncertainty:
        print('Using uncertainty loss!')
        objective = NLLMixtureLaplace()
        loss_module = MultiScaleMixtureDensity(level_weights=weights_level_loss, loss_function=objective, downsample_gt_flow=True)
    else:
        print('Using EPE loss!')
        objective = EPE()
        loss_module = MultiScaleFlow(level_weights=weights_level_loss, loss_function=objective, downsample_gt_flow=True)

    # 6. Define actor
    GLUNetActor = CrocoBasedActor(model, objective=loss_module,batch_processing=batch_processing, nbr_images_to_plot=12, args=args)
    
    # 7. Define Optimizer            
    optimizer = optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=settings.lr, 
                             weight_decay=0.05)
    
    # add more config args to wandb
    if args.log_tool == 'wandb':
        wandb.config.update({'tot_params': tot_params/1e6,
                             'tot_trainable_params': trainable_params/1e6,
                             'tot_model_size': tot_model_size/1e6,
                             'tot_trainable_model_size': trainable_model_size/1e6,
                             'croco_args': croco_args})
        
    # 8. Define Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=settings.scheduler_steps,
                                         gamma=0.5)

    train_val_loader = [train_loader, val_loader]
    # 9. Define Trainer
    trainer = MatchingTrainer(GLUNetActor, train_val_loader, optimizer, settings, lr_scheduler=scheduler, args=args)
    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




