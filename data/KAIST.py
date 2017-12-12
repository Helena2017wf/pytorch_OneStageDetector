"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.augmentations import SSDAugmentation

DATASET_NAME = "KAIST"
CLASSES = ('person',)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

DatasetRoot = "/data/KAIST"

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult

    def _difficult_condition(self,line):

        label = line[0]
        bbox = [int(i) for i in line[1:5]]
        occ = int(line[5])
        ignore = int(line[10])
        vrate = (float(line[9]) * float(line[8])) / ((float(line[3]) + 1e-14) * (float(line[4]) + 1e-14))
        if label != 'person' or bbox[3] < 45 or ignore>0 or bbox[0]<5 or bbox[1]<5 or (bbox[0]+bbox[2])>635 or (bbox[1]+bbox[3])>475:
            return 1
        elif occ > 1 and  vrate <0.65:
            return 1
        else:
            return 0

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        with open(target) as f:
            for line in f:
                line = line.strip().split()
                if line[0] == "%":
                    continue
                else:
                    # difficult = self._difficult_condition(line)
                    # if not difficult or self.keep_difficult:
                    if line[0] == 'person':
                        box = [int(i) for i in line[1:5]]
                        ##bbox format "xywh"
                        ## convert to "xmin,ymin, xmax, ymax"
                        x, y, w, h = box
                        xmin = float(x) / float(width)
                        ymin = float(y) / float(height)
                        xmax = np.minimum(1.0, float(x + w) / float(width))
                        ymax = np.minimum(1.0, float(y + h) / float(height))
                        label_idx = 0 #self.class_to_ind[text]
                        box = [ xmin,ymin, xmax, ymax,label_idx]
                        res += [box]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class GetDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, transform=None, target_transform=None,
                 type='visible',dataset_name='train02_only_person',skip=0):
        self.root = root
        self.type = type
        self.name = dataset_name
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'annotations','%s','%s', '%s.txt')
        self._imgpath = os.path.join('%s', '%s','%s',type,'%s.jpg')
        self.ids = list()
        self.image_names = list()
        for line in open(os.path.join(self.root, 'imageSets', self.name + '.txt')):
            self.ids.append(tuple([self.root]+line.strip().split('/')))
            nn = line.strip().split('/')
            self.image_names.append("{}_{}_{}".format(*line.strip().split('/')))
        if skip:
            self.ids = [x for i, x in enumerate(self.ids) if i%skip==0]
            self.image_names = [x for i, x in enumerate(self.image_names) if i%skip==0]
        self.num_samples = len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def _difficult_condition(self,line):

        label = line[0]
        bbox = [int(i) for i in line[1:5]]
        occ = int(line[5])
        ignore = int(line[10])
        vrate = (float(line[9]) * float(line[8])) / ((float(line[3]) + 1e-14) * (float(line[4]) + 1e-14))
        if label != 'person' or bbox[3] < 45 or ignore>0 or bbox[0]<5 or bbox[1]<5 or (bbox[0]+bbox[2])>635 or (bbox[1]+bbox[3])>475:
            return 1
        elif occ > 1 and  vrate <0.65:
            return 1
        else:
            return 0

    def _anno_parser(self,filename):
        res = []
        with open(filename) as f:
            for line in f:
                line = line.strip().split()
                if line[0] == "%":
                    continue
                else:
                    difficult = self._difficult_condition(line)
                    box = [int(i) for i in line[1:5]]
                    ##bbox format "xywh"
                    ## convert to "xmin,ymin, xmax, ymax"
                    x, y, w, h = box
                    xmin = (x)
                    ymin = (y)
                    xmax = (x + w)
                    ymax = (y + h)
                    label = "person"
                    box = [xmin, ymin, xmax, ymax]
                    info = {'name':label,'bbox':box,'difficult':difficult}
                    res.append(info)
        return res

    def pull_gt_by_class(self,classname):

        class_recs = {}
        npos = 0
        di = 0
        for i in range(self.num_samples):
            id = self.ids[i]
            img_name = self.image_names[i]

            gt = self._anno_parser(self._annopath%id )

            # R = [obj for obj in gt if obj['name'] == classname]
            R = [obj for obj in gt if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            di = di + sum(difficult)
            class_recs[img_name] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
        print (di)
        return class_recs,npos


    def pull_item(self, index):
        img_id = self.ids[index]

        target_path = self._annopath % img_id
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target_path, width, height)


        if len(target) ==0 and self.transform is not None:
            img,_,_ = self.transform(img)
            img = img[:, :, (2, 1, 0)]
            # print(target_path)
        elif self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index,img_width,img_height):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = self._annopath % img_id
        gt = self.target_transform(anno, img_width, img_height)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

if __name__ == "__main__":
    dataset = GetDataset(DatasetRoot,SSDAugmentation(),AnnotationTransform())
