# for server 8
# class EnvironmentSettings:
#     def __init__(self):
#         self.workspace_dir = '/home/cvlab08/projects/data/jinlovespho/dm_final'    # Base directory for saving network checkpoints.
#         self.tensorboard_dir = self.workspace_dir    # Directory for tensorboard files.
#         self.pretrained_networks = self.workspace_dir
#         self.pre_trained_models_dir = ''
#         self.megadepth = ''
#         self.megadepth_csv = ''
#         self.robotcar = ''
#         self.robotcar_csv = ''
#         self.hp = '/home/cvlab08/projects/data/hpatches-sequences-release'
#         self.eth3d = ''
#         self.kitti2012 = ''
#         self.kitti2015 = ''
#         self.sintel = ''
#         self.scannet_test = ''
#         self.yfcc = ''
#         self.tss = ''
#         self.PFPascal = ''
#         self.PFWillow = ''
#         self.spair = ''
#         self.caltech = ''
#         self.training_cad_520 = '/home/cvlab08/projects/data/DPED/DPED'
#         self.validation_cad_520 = '/home/cvlab08/projects/data/DPED/DPED_val'
#         self.coco = '/home/cvlab08/projects/data/COCO2014'
#         self.megadepth_training = '/home/cvlab08/projects/data/MegaDepth'


# for server5
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/dataset1/jinlovespho/dm_final'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir
        self.pre_trained_models_dir = ''
        self.megadepth = ''
        self.megadepth_csv = ''
        self.robotcar = ''
        self.robotcar_csv = ''
        self.hp = '/media/data1/hpatches-sequences-release'
        self.eth3d = ''
        self.kitti2012 = ''
        self.kitti2015 = ''
        self.sintel = ''
        self.scannet_test = ''
        self.yfcc = ''
        self.tss = ''
        self.PFPascal = ''
        self.PFWillow = ''
        self.spair = ''
        self.caltech = ''
        self.training_cad_520 = '/media/dataset2/DPED/DPED'
        self.validation_cad_520 = '/media/dataset2/DPED/DPED_val'
        self.coco = '/media/dataset2/COCO2014'
        self.megadepth_training = '/media/dataset1/MegaDepth'
