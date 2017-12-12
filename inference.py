# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import cv2
import numpy as np
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import CLASSES as labels

from Nets.ssd import build_ssd
# from models import build_ssd as build_ssd_v1 # uncomment for older pool6 model
input_dim = 300

net = build_ssd('test', input_dim, 2)    # initialize SSD

net.load_weights('weights/ssd_300_KAIST_visible_59999.pth')

from matplotlib import pyplot as plt
from data import GetDataset, DatasetRoot, AnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = GetDataset(DatasetRoot, None, AnnotationTransform(),dataset_name='test20')

for index in range(testset.num_samples):
    # if i%10 != 0.0:
    #     continue
    image = testset.pull_image(index)
    img_height,img_width = image.shape[:2]
    _,anno = testset.pull_anno(index,img_width,img_height)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform


    x = cv2.resize(image, (input_dim, input_dim)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)


    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    top_k=10

    # plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.1:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
        currentAxis.text(10, 10, "{}".format(index+1))
    for k in anno:
        coords = (int(k[0]*img_width),int(k[1]*img_height)) , int((k[2]-k[0])*img_width),int((k[3]-k[1])*img_height)
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=2))
    plt.pause(0.001)
    plt.draw()
    plt.gcf().clear()