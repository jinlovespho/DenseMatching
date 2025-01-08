class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir
        self.pre_trained_models_dir = ''
        self.megadepth = ''
        self.megadepth_csv = ''
        self.robotcar = ''
        self.robotcar_csv = ''
        self.hp = '/media/data1/hpatches-sequences-release'
        self.eth3d = '/media/data1/ETH3D'
        self.kitti2012 = ''
        self.kitti2015 = ''
        self.sintel = ''
        self.scannet_test = ''
        self.yfcc = ''
        self.tss = ''
        self.PFPascal = '/media/data1/PF-dataset-PASCAL'
        self.PFWillow = ''
        self.spair = '/media/data1/SPair-71k'
        self.caltech = ''
        self.training_cad_520 = '/media/dataset1/DPED/DPED'
        self.validation_cad_520 = '/media/dataset1/DPED/DPED_val'
        self.coco = '/media/dataset1/COCO2014'
        self.megadepth_training = '/media/dataset1/MegaDepth'
        