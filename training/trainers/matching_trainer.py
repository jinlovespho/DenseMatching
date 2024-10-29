import os
from collections import OrderedDict
from training.trainers.base_trainer import BaseTrainer
from admin.stats import AverageMeter, StatValue
from admin.tensorboard import TensorboardWriter
import torch
import time
import gc

from torchvision.utils import save_image
import torch.nn.functional as F

def warp_image_with_flow(image, flow):
    """
    Warp an image using a flow field.
    
    Args:
        image (torch.Tensor): Image tensor of shape (B, C, H, W)
        flow (torch.Tensor): Flow field tensor of shape (B, 2, H, W)
    
    Returns:
        torch.Tensor: Warped image tensor of shape (B, C, H, W)
    """
    breakpoint()
    # Convert image to float if it's not already
    if image.dtype != torch.float32:
        image = image.float()
    
    B, C, H, W = image.shape
    
    # Create a grid of coordinates
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if image.is_cuda:
        grid = grid.cuda()
    
    # Add flow to the grid
    vgrid = grid + flow
    
    # Normalize grid values to [-1,1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Permute for grid_sample
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Warp image
    output = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return output


class MatchingTrainer(BaseTrainer):
    """Training for matching networks. """
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, make_initial_validation=False, args=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            make_initial_validation - bool, make initial validation before first training epoch ?
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler, make_initial_validation)

        self._set_default_settings()  # set default settings when no values are already set

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # I want to save the checkpoints at the same location than the models and so on
        tensorboard_writer_dir = os.path.join(self._base_save_dir, self.settings.project_path, 'tensorboard')
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.log_tool = args.log_tool

    def _set_default_settings(self):
        # Dict of all default values
        default = {'seed': 500, 'print_interval': 10,
                   'print_stats': None,
                   'description': '',
                   'keep_last_checkpoints': 10,  # keep only the last X checkpoints
                   'dataset_callback_fn': None}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        
        # pbar = tqdm(enumerate(loader), total=len(loader))
        for i, data in enumerate(loader):

            # # vis input data
            # save_image(data['source_image'].float(), f'img_s.jpg', normalize=True)
            # save_image(data['target_image'].float(), f'img_t.jpg', normalize=True)
            # save_image(data['correspondence_mask'].unsqueeze(1).float(), f'img_corr_msk.jpg', normalize=True) 
            # # using data['flow_map'] lets warp the source image to the target image
            # warped_img = warp_image_with_flow(data['source_image'], data['flow_map'])
            # save_image(warped_img.float(), f'img_warped.jpg', normalize=True)   

            # breakpoint()
            data['epoch'] = self.epoch
            data['iter'] = i
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data, loader.training, self.log_tool)

            # backward pass and update weights
            if loader.training:
                grad_is_nan = False
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.actor.net.parameters():
                    if getattr(param, 'grad', None) is not None and ~torch.isfinite(param.grad).all():
                        print('Epoch {}, batch {}, Grad was NAN!!!'.format(self.epoch, i))
                        grad_is_nan = True
                        break
                        # raise Exception('Grad was NAN')
                if not grad_is_nan:
                    self.optimizer.step()

                del loss

            # update statistics
            batch_size = data['source_image'].shape[0]
            self._update_stats(stats, batch_size, loader)
            self._print_stats(i, loader, batch_size)

        if not loader.training:
            # update the current best value, for each epoch, can decide what is the best value.
            self.current_best_val = self.stats[loader.name]['best_value'].avg

    def train_epoch(self):
        """Do one epoch for each loader."""

        if self.epoch == 1 and self.make_initial_validation:
            self.epoch = 0
            self.cycle_dataset(self.loaders[-1])
            self._reset_new_epoch()
            self.epoch = 1

        for loader in self.loaders:
            # do one cycle of training dataset

            # resample the training dataset if dataset_callback_fn exists
            if loader.name == 'train' and self.epoch > 1 and not self.just_started and self.settings.dataset_callback_fn:
                if hasattr(loader.dataset, self.settings.dataset_callback_fn):
                    getattr(loader.dataset, self.settings.dataset_callback_fn)(self.settings.seed + self.epoch)

            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()
        torch.cuda.empty_cache()
        gc.collect()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        if (loader.name == 'train' and (i % self.settings.print_interval == 0 or i == (loader.__len__() - 1))) \
                or (loader.name == 'val' and i == (loader.__len__() - 1)):
            lr = self.lr_scheduler.get_last_lr()[0] if float(torch.__version__[:3]) >= 1.1 else \
                self.lr_scheduler.get_lr()[0]
            print_str = '[%s: epoch %d, batch %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  lr: %.7f   ,  ' % (average_fps, batch_fps, lr)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print('\n' + print_str[:-5])
            if loader.name == 'val':
                print('\n Last validation update {}, value = {}'.format(self.epoch_of_best_val, self.best_val))

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _reset_new_epoch(self):
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        # add all statistics to tensorboard
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
