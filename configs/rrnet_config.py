from datasets.transforms import *
from torch.utils.data import DistributedSampler
from easydict import EasyDict as edict


# Base Config ============================================
Config = edict()
Config.seed = 921
Config.data_root = './data/2019origin'
Config.log_prefix = 'RRNet'
Config.use_tensorboard = True
Config.num_classes = 4

# Training Config =========================================
Config.Train = edict()
# If use the pretrained backbone model.
Config.Train.pretrained = True

# Dataloader params.
Config.Train.batch_size = 4
Config.Train.num_workers = 4
Config.Train.sampler = DistributedSampler

# Optimizer params.
Config.Train.lr = 2.5e-4
Config.Train.momentum = 0.9
Config.Train.weight_decay = 0.0001
# Milestones for changing learning rage.
Config.Train.lr_milestones = [60000, 80000]

Config.Train.iter_num = 100000

# Transforms
Config.mean = (0.238, 0.561, 0.305)
Config.std = (0.085, 0.153, 0.121)
Config.resize_size = (512, 512)
Config.scale_factor = 4

Config.Train.transforms = Compose([
    #MultiScale(scale=(0.8, 0.9, 1, 1.1, 1.2)),
    ResizeBySize(Config.resize_size),
    ToTensor(),
    #RandomCrop(size=(416, 416)),
    HorizontalFlip(),
    Normalize(Config.mean, Config.std),
    ToHeatmap(scale_factor=Config.scale_factor)
])

# Log params.
Config.Train.print_interval = 50
Config.Train.checkpoint_interval = 5000


# Validation Config =========================================
Config.Val = edict()
# Dataloader params.
Config.Val.batch_size = 1
Config.Val.num_workers = 4

Config.result_dir = './results/'


# Model Config ===============================================
Config.Model = edict()

Config.Model.backbone = 'hourglass'
Config.Model.num_stacks = 1
Config.Model.use_rr = True
Config.Model.nms_type_for_stage1 = 'nms'  # or 'soft_nms'
Config.Model.nms_per_class_for_stage1 = True


# Distributed Config =========================================
Config.Distributed = edict()
Config.Distributed.world_size = 1
Config.Distributed.gpu_id = -1
Config.Distributed.rank = 0
Config.Distributed.ngpus_per_node = 1
Config.Distributed.dist_url = 'tcp://127.0.0.1:34564'


Config.is_eval = True
Config.model_path = './log/{}/ckp-59999.pth'.format(Config.log_prefix)
