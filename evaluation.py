"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from Nets import get_net,get_config
from data import AnnotationTransform, GetDataset, BaseTransform, CLASSES,DATASET_NAME
from data import CLASSES as labelmap
from data import DatasetRoot

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--net', default='SSD', help='detection network')
parser.add_argument('--input_dim', default='512', help='the dimension of the input image')
parser.add_argument('--trained_model', default='weights/ssd_512_Sensiac_v_ir_m.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=DatasetRoot, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)

# annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
# YEAR = '2007'
# devkit_path = DatasetRoot + 'VOC' + YEAR
# set_type = 'test'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(args.save_folder, 'python')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_results_file(all_boxes, dataset):
    """
    write the detection results to the text file, class by class
    for each file, the bboxes format is like [image_index, score, xmin, ymin, xmax, ymax]
    :param all_boxes:
    :param dataset:
    :return:
    """
    ###for local python evaluation
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} results file'.format(cls))
        filename = get_results_file_template(DATASET_NAME, cls)
        with open(filename, 'wt') as f:
            for im_ind, name in enumerate(dataset.image_names):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(name, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    ###for matlab pdollar evaluation
    if not os.path.exists("eval/matlab"):
        os.makedirs("eval/matlab")
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} results file'.format(cls))
        filename = "eval/matlab/Dets.txt"
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # [ind, x, y,w,h]
                for k in range(dets.shape[0]):
                    f.write('{:d} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                            format(im_ind+1,
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] - dets[k,0], dets[k, 3] - dets[k,1],
                                   dets[k, -1]*100,))
def do_python_eval(dataset,output_dir='output', use_07=True):
    # cachedir = os.path.join(args.save_folder, 'annotations_cache')
    aps = []
    ams = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_results_file_template(DATASET_NAME, cls)
        tp,fp,npos,nimg = compute_tp_fp(filename,cls,dataset,ovthresh=0.5)
        ap,rec,prec = compute_AP(tp,fp,npos,use_07_metric=True)
        am = compute_MR(tp,fp,nimg,npos)
        aps += [ap]
        ams += [am]
        print('AP for {} = {:.4f}'.format(cls, ap))
        print('log-average miss rate for {} = {:.4f}'.format(cls, am))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Mean log-average miss rate = {:.4f}'.format(np.mean(ams)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))

    map = np.mean(aps)
    mam = np.mean(ams)
    return map,mam


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def average_miss_fppi(miss,fppi):

    # fppi = np.log(fppi)

    # plt.plot(fppi,miss)
    # plt.show()
    am = 0.0
    ref = []
    for t in np.power(10,np.arange(-2,0.25,0.25)):
        if np.sum(fppi<=t) == 0:
            ref.append(0)
        else:
            ref.append(np.min(miss[fppi<=t]))

    log_average = np.exp(np.average(np.log(np.asarray(ref))))
    return log_average

def compute_overlap(dt,gt,ig):
    '''
     Uses modified Pascal criteria with "ignore" regions. The overlap area
    (oa) of a ground truth (gt) and detected (dt) bb is defined as:
     oa(gt,dt) = area(intersect(dt,dt)) / area(union(gt,dt))
    In the modified criteria, a gt bb may be marked as "ignore", in which
    case the dt bb can can match any subregion of the gt bb. Choosing gt' in
    gt that most closely matches dt can be done using gt'=intersect(dt,gt).
    Computing oa(gt',dt) is equivalent to:
    oa'(gt,dt) = area(intersect(gt,dt)) / area(dt)
    :param dt:
    :param gt:
    :param ig:
    :return:
    '''
    n = gt.shape[0]
    overlaps = np.zeros((n,),dtype=np.float64)
    for i in range(n):
        ixmin = np.maximum(gt[i, 0], dt[0])
        iymin = np.maximum(gt[i, 1], dt[1])
        ixmax = np.minimum(gt[i, 2], dt[2])
        iymax = np.minimum(gt[i, 3], dt[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        if ig[i]:
            uni = (dt[2] - dt[0]) * (dt[3] - dt[1])
        else:
            uni = ((dt[2] - dt[0]) * (dt[3] - dt[1]) +
                   (gt[i, 2] - gt[i, 0]) *
                   (gt[i, 3] - gt[i, 1]) - inters)
        oa = inters / uni
        overlaps[i] = oa
    return overlaps

def compute_tp_fp(detpath,
             classname,dataset,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""

    # read gt
    class_recs, npos = dataset.pull_gt_by_class(classname)
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        nimg = dataset.num_samples
        ignore_bbox = 0
        matched = 0
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            # if BBGT.size ==0:
            #     continue
            difficult = R['difficult']
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                # ixmin = np.maximum(BBGT[:, 0], bb[0])
                # iymin = np.maximum(BBGT[:, 1], bb[1])
                # ixmax = np.minimum(BBGT[:, 2], bb[2])
                # iymax = np.minimum(BBGT[:, 3], bb[3])
                # iw = np.maximum(ixmax - ixmin, 0.)
                # ih = np.maximum(iymax - iymin, 0.)
                # inters = iw * ih
                # uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                #        (BBGT[:, 2] - BBGT[:, 0]) *
                #        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                # overlaps = inters / uni
                ###overlaps from matlab ######
                overlaps = compute_overlap(bb,BBGT,difficult)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            #######original metric #########
            # if ovmax > ovthresh:
            #     if not R['difficult'][jmax]:
            #         if not R['det'][jmax]:
            #             tp[d] = 1.
            #             R['det'][jmax] = 1
            #         else:
            #             fp[d] = 1.
            #     else:
            #         ignore_bbox = ignore_bbox + 1
            # else:
            #     fp[d] = 1.

            ##### modified from Matlab version ####
            bstm = 0
            bstg = 0
            bstthr = ovthresh
            for k in range(BBGT.shape[0]):
                if R['det'][k]:
                    continue
                if bstm != 0 and R['difficult'][k]:
                    break
                if overlaps[k] < bstthr:
                    continue
                bstthr = overlaps[k]
                bstg = k
                if R['difficult'][k]:
                    bstm = -1
                    ignore_bbox = ignore_bbox +1
                else:
                    bstm = 1
            if bstm == 0:
                fp[d] = 1.
            elif bstm == 1:
                tp[d] = 1.
                matched = matched +1
                R['det'][bstg] = True

        print("matched numb",matched)
        print("ignore bbox:",ignore_bbox)
        # compute precision recall
        # filter ignore
        # fp_filtered = np.asarray([ x for i,x in enumerate(fp) if x or tp[i]])
        # tp_filtered = np.asarray([x for i,x in enumerate(tp) if x or fp[i]])

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

    # else:
    #     rec = -1.
    #     prec = -1.
    #     ap = -1.
    #     am = -1
    return tp,fp,npos,nimg

def compute_MR(tp,fp,nimg,npos):
    fppi = fp / nimg
    rec = tp / float(npos)
    miss = 1 - rec

    am = 0.0
    ref = []
    for t in np.power(10,np.arange(-2,0.25,0.25)):
        if np.sum(fppi<=t) == 0:
            ref.append(0)
        else:
            ref.append(np.min(miss[fppi<=t]))

    log_average = np.exp(np.average(np.log(np.asarray(ref))))
    return log_average


def compute_AP(tp,fp,npos,use_07_metric):

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap,rec,prec

def ap_eval_old(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers

    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(DATASET_NAME+"_"+args.net+args.input_dim+"_120000", DATASET_NAME)
    det_file = os.path.join(output_dir, 'detections.pkl')

    index = 0
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        # if not len(gt): ### some image dont have gt
        #     continue
        print("%s/%s"%(index,num_images))
        index = index+1
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets
        #all boxes format [classes(2)][num_images][coordinates and score]
        # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
        #                                             num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map, mam = evaluate_detections(all_boxes, output_dir, dataset)
    return map,mam


def evaluate_detections(box_list, output_dir, dataset):
    write_results_file(box_list, dataset)
    map, mam = do_python_eval(dataset,output_dir)

    return map,mam

def run_evaluation(size = None, model_name = None):

    if not model_name:
        model_name = args.trained_model
    if not size:
        size = int(args.input_dim)
    num_classes = len(CLASSES) + 1 # +1 background
    cfg = get_config(args.net+args.input_dim)
    net_class = get_net(args)
    net = net_class('test', size, num_classes,cfg) # initialize SSD
    net.load_state_dict(torch.load(model_name))
    net.eval()
    print('Finished loading model!')
    # load data
    if DATASET_NAME == 'KAIST':
        dataset = GetDataset(args.voc_root, BaseTransform(size, dataset_mean), AnnotationTransform(),dataset_name='test20',skip=0)
    elif DATASET_NAME == 'VOC0712':
        dataset = GetDataset(args.voc_root, BaseTransform(size, dataset_mean), AnnotationTransform(),[('2007','test')])
    elif DATASET_NAME == 'Sensiac':
        dataset = GetDataset(args.voc_root, BaseTransform(size, dataset_mean), AnnotationTransform(),dataset_name='day_test10')
    elif DATASET_NAME == 'Caltech':
        dataset = GetDataset(args.voc_root, BaseTransform(size, dataset_mean), AnnotationTransform(), dataset_name='test01', skip=30)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    map, mam = test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, size,
             thresh=args.confidence_threshold)
    return map, mam

if __name__ == '__main__':

    map, mam = run_evaluation()

    print("map:",map,"mam",mam)
    # dataset = GetDataset(args.voc_root, BaseTransform(300, dataset_mean), AnnotationTransform(), dataset_name='test20',skip=0)
    #
    # do_python_eval(dataset,"ssd300_120000/KAIST/")