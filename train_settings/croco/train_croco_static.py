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
import torch 
import wandb

device= 'cuda' if torch.cuda.is_available() else 'cpu'

def run(settings, args):
    settings.description = 'Train setting for croco on dataset static dataset called DPED-CityScape-ADE'
    settings.data_mode = 'local'
    settings.batch_size = args.batch_size
    settings.n_threads = 8
    settings.multi_gpu = args.multi_gpu
    settings.lr = args.lr
    settings.scheduler_steps = [65, 75, 95]
    settings.n_epochs = args.max_epoch

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

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size, shuffle=True,
                          drop_last=False, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)


    # 3. Define model
    from models.orig_croco.models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
    from models.orig_croco.models.head_downstream import PixelwiseTaskWithDPT

    ckpt = torch.load(args.croco_ckpt, 'cpu')   # crocoflow.pth
    ckpt['args']
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
    model = CroCoDownstreamBinocular(head, **ckpt_args.croco_args)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print('CROCO WEIGHT WELL LOADED: ', msg)
    model.train()
    model = model.to(device)

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model)

    # 4. Define batch_processing
    batch_processing = GLUNetBatchPreprocessing(settings, apply_mask=args.apply_coco_msk, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False)

    # 5, Define loss module
    objective = EPE()
    weights_level_loss = [0.32]
    loss_module = MultiScaleFlow(level_weights=weights_level_loss, loss_function=objective, downsample_gt_flow=True)

    # 6. Define actor
    GLUNetActor = CrocoBasedActor(model, objective=loss_module,batch_processing=batch_processing, args=args)
    

    # 7. Define Optimizer

    if args.freeze_croco_enc:
        print('Freezing encoder parameters!')
        # Freeze parameters
        for name, param in model.named_parameters():
            if 'enc_blocks' in name or 'enc_norm' in name:
                param.requires_grad = False
    else:
        pass
        
            
    optimizer = optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=settings.lr, 
                             weight_decay=0.05)


    total_params = sum(p.numel() for p in model.parameters())
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TOTAL PARAMS: {total_params/1e6:.2f} M")
    print(f"TOTAL PARAMS (TRAINABLE): {total_params_trainable/1e6:.2f} M")
    model_size = sum(p.numel()*p.element_size() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.2f} GB")

    # add more config args to wandb
    if args.log_tool == 'wandb':
        wandb.config.update({'total_params': total_params/1e6,
                             'total_params_trainable': total_params_trainable/1e6,
                             'model_size': model_size/1e6,
                             'croco_args': ckpt_args.croco_args})
        

    # 8. Define Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=settings.scheduler_steps,
                                         gamma=0.5)

    # 9. Define Trainer
    trainer = MatchingTrainer(GLUNetActor, [train_loader, val_loader], optimizer, settings, lr_scheduler=scheduler, args=args)
    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




