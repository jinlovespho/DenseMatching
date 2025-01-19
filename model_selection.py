import os.path as osp
import torch
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_model(model_name, path_to_pre_trained_models, args):

    estimate_uncertainty = False
    
    if model_name == 'crocov2':
        from models.croco.croco import CroCoNet
        from models.croco.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular

        estimate_uncertainty = args.uncertainty

        ckpt = torch.load(args.croco_ckpt,'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = ((args.model_img_size[0]//32)*32,(args.model_img_size[1]//32)*32)
        croco_args['args'] = args
        network = CroCoNet(**croco_args)
        msg=network.load_state_dict(ckpt['model'], strict=False)
        print('missing keys: ', msg.missing_keys) # model 에는 있는데 ckpt 에는 없는 것들
        print('unexpected keys: ', msg.unexpected_keys) # ckpt 에는 있는데 model 에는 없는 것들
        print('CROCOV2 WEIGHT WELL LOADED: ', msg)
        network.eval()
        network = network.to(device)
    
    elif model_name == 'dift_sd':
        from models.dift.dift_sd import SDFeaturizer4Eval
        all_cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
        network = SDFeaturizer4Eval(cat_list=all_cats)
    # elif model_name == 'sd3_single':
    #     from models.sd3_single.sd3_single import SD3Single
    #     network = SD3Single(args)
    elif model_name == 'sd3_single' or model_name == 'sd3_joint':
        from models.sd3_joint.sd3_joint import SD3Joint
        network = SD3Joint(args)      
    elif model_name == 'dit_single':
        from models.dit_single.dit_single import DITSingle
        network = DITSingle(args)
    elif model_name == 'cogvid_single':
        from models.cogvid_single.cogvid_single import CogVidSingle
        network = CogVidSingle(args)
    
    else:
        print('ERROR!!!! Model Name: ', model_name)

    if path_to_pre_trained_models is not None:
        if path_to_pre_trained_models.endswith('.pth') or path_to_pre_trained_models.endswith('.pth.tar') or path_to_pre_trained_models.endswith('.pt'):    # true
            # if the path already corresponds to a checkpoint path, we use it directly
            checkpoint_fname = path_to_pre_trained_models
        network = load_network(network, checkpoint_path=checkpoint_fname)
        network.eval()
        network = network.to(device)
    else:
        print('No pre-trained model path provided')

    return network, estimate_uncertainty
