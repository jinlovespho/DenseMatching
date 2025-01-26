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
from training.actors.self_supervised_actor import CrocoBasedActor
from models.GLUNet.GLU_Net import glunet_vgg16
from datasets.load_pre_made_datasets.load_pre_made_dataset import PreMadeDataset

# JLP
import os
import torch 
import wandb
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import datetime

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
    img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    co_transform = None

    # base training data is DPED-CityScape-ADE
    train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                      source_image_transform=img_transforms,
                                      target_image_transform=img_transforms,
                                      flow_transform=flow_transform,
                                      co_transform=co_transform,
                                      split=1,
                                      get_mapping=False,
                                      img_size=args.img_size)   # resize images to args.img_size

    # validation dataset
    _, val_dataset = PreMadeDataset(root=settings.env.validation_cad_520,
                                    source_image_transform=img_transforms,
                                    target_image_transform=img_transforms,
                                    flow_transform=flow_transform,
                                    co_transform=co_transform,
                                    split=0,
                                    img_size=args.img_size)
    
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

        # finetune crocoflow on DPED from crocov2 ckpt
        if args.croco_ckpt is not None:
            croco_ckpt = torch.load(args.croco_ckpt, 'cpu')
            crocoflow_ckpt = torch.load(args.crocoflow_ckpt, 'cpu')
            crocoflow_ckpt['args'].croco_args['img_size'] = args.img_size   # 224 224  
            crocoflow_ckpt['args'].crop = args.img_size                     # 224 224 
            crocoflow_ckpt['model'] = croco_ckpt['model']
            
        # finetune crocoflow on DPED from crocoflow ckpt
        else:
            crocoflow_ckpt = torch.load(args.crocoflow_ckpt, 'cpu')
        
        # ckpt = torch.load(args.croco_ckpt, 'cpu')   # crocoflow.pth
        # ckpt_args = ckpt['args'] if 'args' in ckpt.keys() else ckpt['croco_kwargs']
        # ckpt_args.croco_args['img_size'] = args.img_size if args.img_size is not None else [320, 384]
        task = crocoflow_ckpt['args'].task   # 'flow'
        num_channels = {'stereo': 1, 'flow': 2}[task]   # 2
        with_conf = True
        if with_conf: num_channels += 1
        print('head: PixelwiseTaskWithDPT()')
        head = PixelwiseTaskWithDPT()
        head.num_channels = num_channels
        print('croco_args:', crocoflow_ckpt['args'].croco_args)
        croco_args = crocoflow_ckpt['args'].croco_args
        croco_args['args'] = args
        model = CroCoDownstreamBinocular(head, **croco_args)
        msg = model.load_state_dict(crocoflow_ckpt['model'], strict=False)
        if dist.get_rank() == 0:
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
        if dist.get_rank() == 0:
            print('CROCO WEIGHT WELL LOADED: ', msg)
        model = model.to(device)
        model.train()
        
    else:
        raise NotImplementedError(f'Model {args.model} not selected!')

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
    tot_params = sum(p.numel() for p in model.parameters()) 
    tot_model_size = sum(p.numel()*p.element_size() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_model_size = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
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
                                                sparse_ground_truth=False)

    # 5, Define loss module
    objective = EPE()
    if args.model == 'crocoflow':
        weights_level_loss = [0.32]
    elif args.model == 'croco_catseg':
        if args.without_catseg_up:
            weights_level_loss = [0.32]
        else: 
            weights_level_loss = [0.32, 0.32]
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
    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




