from termcolor import colored
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler


from utils_data.image_transforms import ArrayToTensor
from training.actors.batch_processing import GLUNetBatchPreprocessing
from training.losses.basic_losses import EPE
from training.losses.multiscale_loss import MultiScaleFlow
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from admin.multigpu import MultiGPU
from datasets.load_pre_made_datasets.load_pre_made_dataset import PreMadeDataset
from datasets.object_augmented_dataset import MSCOCO, AugmentedImagePairsDatasetMultipleObjects
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import RandomAffine

# JLP
from training.actors.self_supervised_actor import CrocoBasedActor
import torch 
import wandb
import os
import os.path as osp

device= 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print(msg)
    print('---------------------------------------')
    print('Weight Loaded from .tar !')
    print('Checkpoint Path: ', checkpoint_path)
    print('missing keys: ', msg.missing_keys) # model 에는 있는데 ckpt 에는 없는 것들
    print('unexpected keys: ', msg.unexpected_keys) # ckpt 에는 있는데 model 에는 없는 것들
    print('---------------------------------------')
    return net

def run(settings, args):
    settings.description = 'Default train settings for GLU-Net on the dynamic dataset (from GOCor paper)'
    settings.data_mode = 'local'
    settings.batch_size = args.batch_size
    settings.n_threads = 8
    settings.multi_gpu = args.multi_gpu
    settings.print_interval = 500
    settings.lr = args.lr
    settings.scheduler_steps = [65, 75, 95]
    settings.n_epochs = args.max_epoch

    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    co_transform = None

    # geometric transformation for moving objects
    fg_tform = RandomAffine(p_flip=0.0, max_rotation=30.0,
                            max_shear=0, max_ar_factor=0.,
                            max_scale=0.3, pad_amount=0)

    # object dataset
    min_target_area = 1300
    coco_dataset_train = MSCOCO(root=settings.env.coco, split='train', version='2014',
                                min_area=min_target_area)

    # base training data is DPED-CityScape-ADE + 1 object from COCO
    train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                      source_image_transform=None,
                                      target_image_transform=None,
                                      flow_transform=None,
                                      co_transform=None,
                                      split=1)  # only training

    # we then adds the object on the dataset
    train_dataset = AugmentedImagePairsDatasetMultipleObjects(foreground_image_dataset=coco_dataset_train,
                                                              background_image_dataset=train_dataset,
                                                              foreground_transform=fg_tform,
                                                              number_of_objects=1, object_proba=0.8,
                                                              source_image_transform=img_transforms,
                                                              target_image_transform=img_transforms,
                                                              flow_transform=flow_transform,
                                                              co_transform=co_transform)

    # validation dataset: DPED-CityScape-ADE + 1 object from COCO
    _, val_dataset = PreMadeDataset(root=settings.env.validation_cad_520,
                                    source_image_transform=None,
                                    target_image_transform=None,
                                    flow_transform=None,
                                    co_transform=None,
                                    split=0)

    val_dataset = AugmentedImagePairsDatasetMultipleObjects(foreground_image_dataset=coco_dataset_train,
                                                            background_image_dataset=val_dataset,
                                                            number_of_objects=1, object_proba=0.8,
                                                            foreground_transform=fg_tform,
                                                            source_image_transform=img_transforms,
                                                            target_image_transform=img_transforms,
                                                            flow_transform=flow_transform,
                                                            co_transform=co_transform)

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size, shuffle=True,
                          drop_last=False, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)

    # 3. Define model
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
        print('head: PixelwiseTaskWithDPT()')
        head = PixelwiseTaskWithDPT()
        head.num_channels = num_channels
        print('croco_args:', ckpt_args.croco_args)
        croco_args = ckpt_args.croco_args
        model = CroCoDownstreamBinocular(head, **croco_args)
        msg = model.load_state_dict(ckpt['model'], strict=True)
        print('CROCO WEIGHT WELL LOADED: ', msg)
        model.train()
        model = model.to(device)        

    elif args.model =='croco_catseg':
        from models.croco.croco import CroCoNet
        from models.croco.croco_downstream import croco_args_from_ckpt

        ckpt = torch.load(args.croco_ckpt, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = args.img_size
        croco_args['args'] = args
        model = CroCoNet(**croco_args)
        msg=model.load_state_dict(ckpt['model'], strict=False)
        print('CROCO WEIGHT WELL LOADED: ', msg)
        model.train()
        model = model.to(device)
    
    elif args.model == 'crocov2':
        from models.croco.croco import CroCoNet
        from models.croco.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular

        # weights_already_loaded = True
        estimate_uncertainty = args.uncertainty

        # breakpoint()
        ckpt = torch.load(args.croco_ckpt,'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = ((args.img_size[0]//32)*32,(args.img_size[1]//32)*32)
        croco_args['args'] = args
        model = CroCoNet(**croco_args)
        msg=model.load_state_dict(ckpt['model'], strict=False)
        print('missing keys: ', msg.missing_keys) # model 에는 있는데 ckpt 에는 없는 것들
        print('unexpected keys: ', msg.unexpected_keys) # ckpt 에는 있는데 model 에는 없는 것들
        print('CROCOV2 WEIGHT WELL LOADED: ', msg)
        model.train()
        model = model.to(device)

    else:
        raise NotImplementedError(f'Model {args.model} not implemented')

    path_to_pre_trained_models = args.path_to_pre_trained_models
    if path_to_pre_trained_models is not None:
        if path_to_pre_trained_models.endswith('.pth') or path_to_pre_trained_models.endswith('.pth.tar') or path_to_pre_trained_models.endswith('.pt'):    # true
            # if the path already corresponds to a checkpoint path, we use it directly
            checkpoint_fname = path_to_pre_trained_models
        else:   # false
            # it is the path to the directory containing all checkpoints.
            checkpoint_fname = osp.join(path_to_pre_trained_models, model_name + '_{}'.format(pre_trained_model_type) + '.pth')
            if not os.path.exists(checkpoint_fname):
                checkpoint_fname = checkpoint_fname + '.tar'
        model = load_network(model, checkpoint_path=checkpoint_fname)
        model.train()
        model = model.to(device)
    else:
        print('No pre-trained model path provided')
    
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
    
    # 3-3. Show params and trainable params
    print('----------------------------------------------------------------')   
    tot_params = sum(p.numel() for p in model.parameters()) 
    tot_model_size = sum(p.numel()*p.element_size() for p in model.parameters()) 
    print(f"TOTAL PARAMS: {tot_params/1e6:.2f} M, TOTAL MODEL SIZE: {tot_model_size/1e6:.2f} MB")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_model_size = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    print(f"TRAINABLE PARAMS: {trainable_params/1e6:.2f} M, TRAINABLE MODEL SIZE: {trainable_model_size/1e6:.2f} MB")
    print('----------------------------------------------------------------')
    
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model)

    # 4. Define batch_processing
    batch_processing = GLUNetBatchPreprocessing(settings, apply_mask=args.apply_coco_msk, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False)

    # 5, Define loss module
    objective = EPE()
    if args.model == 'crocoflow':
        weights_level_loss = [0.32]
    elif args.model == 'croco_catseg':
        weights_level_loss = [0.32, 0.32]
    elif args.model == 'crocov2':
        weights_level_loss = [0.32]
    loss_module = MultiScaleFlow(level_weights=weights_level_loss, loss_function=objective, downsample_gt_flow=True)

    # 6. Define actor
    GLUNetActor = CrocoBasedActor(model, objective=loss_module,batch_processing=batch_processing, nbr_images_to_plot=6, args=args)
    

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
    trainer.train(settings.n_epochs, load_latest=False, fail_safe=True)







