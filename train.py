import argparse
import os
import time

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import evaluation
from logger import Logger
from Nets import get_net,get_config
from data import DatasetRoot,GetDataset,AnnotationTransform,CLASSES,detection_collate,DATASET_NAME
from layers.modules import MultiBoxLoss
from utils.augmentations import SSDAugmentation


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--net', default='PDN', help='detection network')
parser.add_argument('--input_dim', default='512', help='the dimension of the input image')
parser.add_argument('--img_type', default='visible', help='format of image (visible, lwir,...)')
parser.add_argument('--log_dir', default='./log', help='the path for saving log infomation')
parser.add_argument('--log_step', default=10, type=int, help='the step for printing log infomation')
parser.add_argument('--save_images_step', default=500, type=int, help='the step for saving images')
parser.add_argument('--model_save_step', default=2000, type=int, help='the step for saving model')

parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--resume', default="weights/PDN512_Caltech_visible_5999.pth", type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--step_values', default=(80000,100000), type=list, help='the steps for decay learning rate')
parser.add_argument('--start_iter', default=6000, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')

parser.add_argument('--send_images_to_tensorboard', type=str2bool, default=True, help='Sample a random image from each log batch,'
                                                                                      ' send it to tensorboard after augmentations step')
parser.add_argument('--validation', type=str2bool, default=True, help='validate the trained model')
# parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
# parser.add_argument('--send_images_to_visdom', type=str2bool, default=True, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=DatasetRoot, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = get_config(args.net+args.input_dim)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

log_dir = os.path.join(args.log_dir,args.net+args.input_dim+"_"+DATASET_NAME)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = Logger(log_dir)

image_size = int(args.input_dim)  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(CLASSES) + 1
batch_size = args.batch_size

######## For future multi-GPU training #########
# accum_batch_size = 32
# iter_size = accum_batch_size / batch_size

# max_iter = 120000
# weight_decay = 0.0005
# stepvalues = (80000, 100000, 120000)
# gamma = 0.1
# momentum = 0.9

# if args.visdom:
#     import visdom
#     viz = visdom.Visdom()

net_class = get_net(args.net)
net = net_class('train', image_size, num_classes,cfg)
parallel_net = net
print_network(net,args.net+args.input_dim)
if args.cuda:
    parallel_net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    net._vgg.load_state_dict(vgg_weights)

if args.cuda:
    parallel_net = parallel_net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    if net._extras:
        net._extras.apply(weights_init)
    net._loc.apply(weights_init)
    net._conf.apply(weights_init)

optimizer = optim.SGD(parallel_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg, num_classes, image_size, 0.35, True, 0, True, 3, 0.5, False, args.cuda)



def train():
    parallel_net.train()
    # loss counters
    # loc_loss = 0  # epoch
    # conf_loss = 0
    # epoch = 0:
    print('Loading Dataset...')
    dataset = GetDataset(args.voc_root, SSDAugmentation(
        image_size, means,type=args.img_type), AnnotationTransform(),type=args.img_type)

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, args.iterations):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in args.step_values:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            # loc_loss = 0
            # conf_loss = 0
            # epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = parallel_net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]
        if iteration % args.log_step == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            logger.scalar_summary("bbox_regression_loss",loss_l.data[0],iteration)
            logger.scalar_summary("classification_loss",loss_c.data[0],iteration)
            logger.scalar_summary("total_loss",loss.data[0],iteration)

            if args.send_images_to_tensorboard and iteration % args.save_images_step == 0:
                logger.image_summary("agumentation images",images.data.cpu().numpy(),iteration)

        if (iteration+1) % args.model_save_step == 0:
            print('Saving state, iter:', iteration)
            save_path = 'weights/' + args.net+args.input_dim +'_'+ DATASET_NAME + "_" + args.img_type + "_" + repr(iteration) + '.pth'
            torch.save(net.state_dict(), save_path )

            if args.validation:
            ####evaluation##########
                print("runing evaluation!!!!")
                map, mam = evaluation.run_evaluation(input_dim=image_size,net_name= args.net, saved_model_name=save_path,skip=300)
                logger.scalar_summary("mAP",map,iteration)
                logger.scalar_summary("Average_Missing_Rate",mam,iteration)

    torch.save(net.state_dict(), args.save_folder + args.net+args.input_dim +'_'+ DATASET_NAME + "_" + args.img_type + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
