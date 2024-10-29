from termcolor import colored
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch

from utils_data.image_transforms import ArrayToTensor
from training.actors.batch_processing import GLUNetBatchPreprocessing
from training.losses.basic_losses import EPE
from training.losses.multiscale_loss import MultiScaleFlow
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from admin.multigpu import MultiGPU
from training.actors.self_supervised_actor import GLUNetBasedActor, CrocoBasedActor
from datasets.load_pre_made_datasets.load_pre_made_dataset import PreMadeDataset
from datasets.object_augmented_dataset import MSCOCO, AugmentedImagePairsDatasetMultipleObjects
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import RandomAffine
## hg add
from models.croco.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular
from models.croco.croco import CroCoNet
from models.croco.head_downstream import PixelwiseTaskWithDPT
from models.croco.pos_embed import interpolate_pos_embed
from training.losses.neg_log_likelihood import NLLMixtureLaplace
from training.losses.multiscale_loss import MultiScaleMixtureDensity


def run(settings, args=None):
    settings.description = 'Default train settings for GLU-Net on the dynamic dataset (from GOCor paper)'
    settings.data_mode = 'local'
    settings.batch_size = args.batch_size #24
    settings.n_threads = 8
    settings.multi_gpu = False
    settings.print_interval = 100
    settings.lr = 0.0001 if args.lr == None else args.lr
    settings.scheduler_steps = [100, 120, 130]
    settings.n_epochs = 150
    
    # 1. Define training and validation datasets
    # datasets, pre-processing of the images is done within the network function !
    img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    co_transform = None

    train_dataset, _ = PreMadeDataset(root=settings.env.training_cad_520,
                                      source_image_transform=img_transforms,
                                      target_image_transform=img_transforms,
                                      flow_transform=flow_transform,
                                      co_transform=co_transform,
                                      split=1,
                                      get_mapping=False, img_size=args.img_size)

    # validation dataset
    _, val_dataset = PreMadeDataset(root=settings.env.validation_cad_520,
                                    source_image_transform=img_transforms,
                                    target_image_transform=img_transforms,
                                    flow_transform=flow_transform,
                                    co_transform=co_transform,
                                    split=0, img_size=args.img_size)

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size, shuffle=True,
                          drop_last=False, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)
    
    if args.eval_ds is not None:
        # EVAL dataset 
        import os 
        import datasets
        from torch.utils.data import DataLoader
        from validation.flow_evaluation.evaluate_per_dataset import (run_evaluation_generic, run_evaluation_kitti,
                                                                run_evaluation_megadepth_or_robotcar,
                                                                run_evaluation_sintel, run_evaluation_eth3d,
                                                                run_evaluation_semantic, run_evaluation_caltech)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        target_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
        input_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first
        estimate_uncertainty = True if args.uncertainty is not None else False

        path_to_save = '/media/dataset3/jinlovespho/log/DenseMatching'
        
        if args.eval_ds == 'megadepth':
            output = run_evaluation_megadepth_or_robotcar(network, settings.env.megadepth,
                                                            path_to_csv=settings.env.megadepth_csv,
                                                            estimate_uncertainty=estimate_uncertainty,
                                                            path_to_save=path_to_save, plot=args.plot,
                                                            plot_100=args.plot_100,
                                                            plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'robotcar':
            output = run_evaluation_megadepth_or_robotcar(network, settings.env.robotcar,
                                                            path_to_csv=settings.env.robotcar_csv,
                                                            estimate_uncertainty=estimate_uncertainty,
                                                            path_to_save=path_to_save, plot=args.plot,
                                                            plot_100=args.plot_100,
                                                            plot_ind_images=args.plot_individual_images)

        elif 'hp' in args.eval_ds:
            original_size = True
            if args.eval_ds == 'hp-240':
                original_size = False
            number_of_scenes = 5 + 1
            list_of_outputs = []
            # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
            for id, k in enumerate(range(2, number_of_scenes + 2)):
                if id == 5:
                    _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                            os.path.join('assets',
                                                                        'hpatches_all.csv'.format(k)),
                                                            input_transform, target_transform, co_transform,
                                                            use_original_size=original_size, split=0)
                else:
                    _, test_set = datasets.HPatchesdataset(settings.env.hp,
                                                            os.path.join('assets',
                                                                        'hpatches_1_{}.csv'.format(k)),
                                                            input_transform, target_transform, co_transform,
                                                            use_original_size=original_size, split=0)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                output_scene = run_evaluation_generic(network, test_dataloader, device,
                                                        estimate_uncertainty=estimate_uncertainty)
                list_of_outputs.append(output_scene)

            output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2],
                        'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[5]}

        elif args.eval_ds == 'kitti2012':
            _, test_set = datasets.KITTI_occ(settings.env.kitti2012, source_image_transform=input_transform,
                                                target_image_transform=input_transform,
                                                flow_transform=target_transform, co_transform=co_transform, split=0)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_kitti(network, test_dataloader, device,
                                            estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                            plot=args.plot, plot_100=args.plot_100,
                                            plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'kitti2015':
            _, test_set = datasets.KITTI_occ(settings.env.kitti2015, source_image_transform=input_transform,
                                                target_image_transform=input_transform,
                                                flow_transform=target_transform, co_transform=co_transform, split=0)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)

            output = run_evaluation_kitti(network, test_dataloader, device,
                                            estimate_uncertainty=estimate_uncertainty, path_to_save=path_to_save,
                                            plot=args.plot, plot_100=args.plot_100,
                                            plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'TSS':
            output = {}
            for sub_data in ['FG3DCar', 'JODS', 'PASCAL']:
                path_to_save_ = os.path.join(path_to_save, sub_data)
                if not os.path.exists(path_to_save_) and (args.plot or args.plot_100):
                    os.makedirs(path_to_save_)
                test_set = datasets.TSSDataset(os.path.join(settings.env.tss, sub_data),
                                                source_image_transform=input_transform,
                                                target_image_transform=input_transform, flow_transform=target_transform,
                                                co_transform=co_transform)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                results = run_evaluation_semantic(network, test_dataloader, device,
                                                    estimate_uncertainty=estimate_uncertainty,
                                                    flipping_condition=args.flipping_condition,
                                                    path_to_save=path_to_save_, plot=args.plot, plot_100=args.plot_100,
                                                    plot_ind_images=args.plot_individual_images)
                output[sub_data] = results

        elif args.eval_ds == 'PFPascal':
            test_set = datasets.PFPascalDataset(settings.env.PFPascal, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                                flipping_condition=args.flipping_condition,
                                                path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                                plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'PFWillow':
            test_set = datasets.PFWillowDataset(settings.env.PFWillow, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device,
                                                estimate_uncertainty=estimate_uncertainty,
                                                flipping_condition=args.flipping_condition,
                                                path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                                plot_ind_images=args.plot_individual_images)
        elif args.eval_ds == 'spair':
            test_set = datasets.SPairDataset(settings.env.spair, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_semantic(network, test_dataloader, device,
                                                estimate_uncertainty=estimate_uncertainty,
                                                flipping_condition=args.flipping_condition,
                                                path_to_save=path_to_save, plot=args.plot, plot_100=args.plot_100,
                                                plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'caltech':
            test_set = datasets.CaltechDataset(settings.env.caltech, source_image_transform=input_transform,
                                                target_image_transform=input_transform, split='test',
                                                flow_transform=target_transform)
            test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
            output = run_evaluation_caltech(network, test_dataloader, device, estimate_uncertainty=estimate_uncertainty,
                                            flipping_condition=args.flipping_condition, path_to_save=path_to_save,
                                            plot_ind_images=args.plot_individual_images)

        elif args.eval_ds == 'sintel':
            output = {}
            for dstype in ['clean', 'final']:
                _, test_set = datasets.mpi_sintel(settings.env.sintel, source_image_transform=input_transform,
                                                    target_image_transform=input_transform,
                                                    flow_transform=target_transform, co_transform=co_transform, split=0,
                                                    load_occlusion_mask=True, dstype=dstype)
                test_dataloader = DataLoader(test_set, batch_size=1, num_workers=8)
                results = run_evaluation_sintel(network, test_dataloader, device,
                                                estimate_uncertainty=estimate_uncertainty)
                output[dstype] = results

        elif args.eval_ds == 'eth3d':
            output = run_evaluation_eth3d(network, settings.env.eth3d, input_transform, target_transform, co_transform,
                                            device, estimate_uncertainty=estimate_uncertainty)

        else:
            raise ValueError('Unknown EVAL dataset, {}'.format(args.eval_ds))
        

    # 3. Define model
    ckpt = torch.load(settings.env.croco_pretrained_path,'cpu')
    croco_args = croco_args_from_ckpt(ckpt)
    croco_args['img_size'] = ((224//32)*32,(224//32)*32)
    croco_args['args'] = args

    model = CroCoNet(**croco_args)
    model.load_state_dict(ckpt['model'], strict=False)
    
    if not args.not_freeze:
        if args.cost_agg == 'cats':
            for key,value in model.named_parameters():
                if 'cats' not in key:
                    value.requires_grad = False
        elif args.cost_agg == 'CRAFT':
            for key,value in model.named_parameters():
                if 'craft' not in key:
                    value.requires_grad = False
        elif args.cost_agg == 'hierarchical_cats' or args.cost_agg == 'hierarchical_residual_cats' or args.cost_agg == 'hierarchical_conv4d_cats' or args.cost_agg == 'hierarchical_conv4d_cats_level'or args.cost_agg == 'hierarchical_conv4d_cats_level_4stage':
            for key,value in model.named_parameters():
                if 'cats' not in key:
                    value.requires_grad = False
        elif args.cost_agg == 'croco_flow':
            for key,value in model.named_parameters():
                if 'head' not in key:
                    value.requires_grad = False

    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model)

    # 4. Define batch_processing
    batch_processing = GLUNetBatchPreprocessing(settings, apply_mask=False, apply_mask_zero_borders=False,
                                                sparse_ground_truth=False)
    # 5, Define loss module
    weights_level_loss = [0.32, 0.08, 0.02, 0.01]
    
    if args.uncertainty:
        objective = NLLMixtureLaplace()
        weights_level_loss = [0.32, 0.08, 0.02, 0.01]
        loss_module_256 = MultiScaleMixtureDensity(level_weights=[0.32], loss_function=objective,
                                        downsample_gt_flow=True)
        loss_module = MultiScaleFlow(level_weights=[0.32], loss_function=objective, downsample_gt_flow=True)
    elif args.hierarchical: # true
        objective = EPE()
        if args.cost_agg == 'hierarchical_cats' or args.cost_agg == 'hierarchical_conv4d_cats' or args.cost_agg == 'hierarchical_residual_cats' or args.cost_agg == 'hierarchical_conv4d_cats_level':
            weights_level_loss = [0.32, 0.32, 0.32, 0.32,0.32,0.32]
        elif args.cost_agg == 'hierarchical_conv4d_cats_level_4stage':
            weights_level_loss = [0.32, 0.16, 0.08, 0.04, 0.02] if args.hierarchical_weights else [0.32] * 5
        loss_module_256 = MultiScaleFlow(level_weights=weights_level_loss[:2], loss_function=objective,
                                    downsample_gt_flow=True)
        loss_module = MultiScaleFlow(level_weights=weights_level_loss, loss_function=objective, downsample_gt_flow=True)
    else:
        objective = EPE()
        weights_level_loss = [0.32, 0.08, 0.02, 0.01]
        loss_module_256 = MultiScaleFlow(level_weights=weights_level_loss[:2], loss_function=objective,
                                                downsample_gt_flow=True)
        loss_module = MultiScaleFlow(level_weights=[0.32], loss_function=objective,
                                            downsample_gt_flow=True)

    # 6. Define actor
    CrocoActor = CrocoBasedActor(model, objective=loss_module, objective_256=loss_module_256,
                                   batch_processing=batch_processing, cost_agg = True)

    # 7. Define Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=settings.lr,
                   weight_decay=0.0004)
    # 8. Define Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=settings.scheduler_steps,
                                         gamma=0.5)

    # 9. Define Trainer
    trainer = MatchingTrainer(CrocoActor, [train_loader, val_loader], optimizer, settings, lr_scheduler=scheduler)

    trainer.train(settings.n_epochs, load_latest=args.load_latest, fail_safe=True)
