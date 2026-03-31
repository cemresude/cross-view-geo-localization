import os
import torch
import yaml
from model import two_view_net, three_view_net
from model_rgbd import two_view_net_rgbd

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./model', name)
    last_model_name = get_model_list(dirname, 'net')
    
    if last_model_name is None:
        raise FileNotFoundError(f"No model found in {dirname}")
    
    # Use the full path directly instead of reconstructing it
    model_path = last_model_name
    
    # Extract epoch for return value
    epoch_str = os.path.basename(last_model_name).split('_')[1]
    epoch = epoch_str.split('.')[0]
    if epoch != 'last':
        try:
            epoch = int(epoch)
        except ValueError:
            epoch = epoch  # Keep as string if not a number
    
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config.get('use_dense', False)
    opt.fp16 = config.get('fp16', False)
    opt.views = config['views']

    # RGBD support
    opt.use_rgbd = config.get('use_rgbd', False)
    
    # VGG16 support
    opt.use_vgg16 = config.get('use_vgg16', False)

    # LPN support
    opt.lpn_blocks = config.get('lpn_blocks', 4)
    opt.lpn_mode = config.get('lpn_mode', 'square')

    # Build LPN kwargs
    lpn_kwargs = {}
    if opt.pool == 'lpn':
        lpn_kwargs = {'lpn_blocks': opt.lpn_blocks, 'lpn_mode': opt.lpn_mode}

    model = None

    if opt.use_rgbd and opt.views == 2:
        # RGBD two-view model
        print("🌈 Loading RGBD two-view model")
        model = two_view_net_rgbd(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, **lpn_kwargs)
    elif opt.views == 2:
        # Standard RGB two-view model
        model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, VGG16=opt.use_vgg16, **lpn_kwargs)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, VGG16=opt.use_vgg16)

    # Load weights
    print(f"📂 Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return model, opt, epoch

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)

