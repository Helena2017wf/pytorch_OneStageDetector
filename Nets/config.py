# config.py
import os.path

# # gets home dir cross platform
# # home = os.path.expanduser("~")
# ddir = os.path.join("/data/PASCAL_VOC","VOCdevkit/")
#
# # note: if you used our download scripts, this should be right
# VOCroot = ddir # path to VOCdevkit root dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
SSD512 = {

    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
             512, 512, 512],
    'extra': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128,'S',256, 128,'L'],
    'mbox': [4, 6, 6, 6, 6, 4, 4],

    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[1,2], [1,2, 3], [1,2, 3], [1, 2, 3],[1, 2,3],[1,2], [1,2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'SSD512',
}

SSD300 = {

    'base':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    'extra':[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'mbox': [4, 6, 6, 6, 4, 4],


    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],   ##### h/w  this is for VOC
    ##TODO
    'aspect_ratios': [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2]],  ##### h/w   this is for pedestrain detection    RPN+BDT use the 2.5
    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'SSD300',
}

PDN512 = {
    ###scales: 0.04,0.155,0.27,0.385.0.5
    'base':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],

    'mbox':[2, 2, 2, 2],

    'feature_maps' : [128, 64, 32, 16],

    'min_dim' : 512,

    'steps' : [4,8, 16, 32],

    'min_sizes' : [16, 32, 64, 128],

    'max_sizes' : [32, 64, 128, 256],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[3], [3], [3], [3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'PDN512',
}

config_factory = {
    'SSD300': SSD300,
    'SSD512': SSD512,
    'PDN512': PDN512,
}
def get_config(name):
    return config_factory[name]

# use average pooling layer as last layer before multibox layers
# v1 = {
#     'feature_maps' : [38, 19, 10, 5, 3, 1],
#
#     'min_dim' : 300,
#
#     'steps' : [8, 16, 32, 64, 100, 300],
#
#     'min_sizes' : [30, 60, 114, 168, 222, 276],
#
#     'max_sizes' : [-1, 114, 168, 222, 276, 330],
#
#     # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
#     'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
#                         [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],
#
#     'variance' : [0.1, 0.2],
#
#     'clip' : True,
#
#     'name' : 'v1',
# }
