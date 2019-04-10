# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""

import io
import argparse
import pprint
import flask
import torch
import cv2
import os
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms as T
import numpy as np
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.blob import im_list_to_blob
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.net_utils import vis_detections

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="./models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--result_dir', dest='result_dir',
                        help='directory to load images for demo',
                        default="./result")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=9449, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def load_model(args):
    """Load the pre-trained model, you can use your model just as easily.

    """

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    # print('Using config:')
    # pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # pascal_classes = np.asarray(['__background__',
    #                              'aeroplane', 'bicycle', 'bird', 'boat',
    #                              'bottle', 'bus', 'car', 'cat', 'chair',
    #                              'cow', 'diningtable', 'dog', 'horse',
    #                              'motorbike', 'person', 'pottedplant',
    #                              'sheep', 'sofa', 'train', 'tvmonitor'])

    global pascal_classes
    pascal_classes = np.asarray(['__background__', 'gangjin'])

    # initilize the network here.
    global fasterRCNN
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)

    fasterRCNN.create_architecture()

    # print("load checkpoint %s" % load_name)
    if use_gpu:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')


def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            box_re = raber_detection(np.array(image))

            data['predictions'] = list()
            l = {"BoxNumber": box_re.shape[0]}
            data['predictions'].append(str(l))
            # Loop over the results and add them to the list of returned predictions
            for box in box_re:
                r = {"BoxList": [str(i) for i in np.rint(box[:4]).astype(int)]}
                data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


def raber_detection(im_in):
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if use_gpu:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    thresh = 0.05
    vis = False

    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :, ::-1]

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()

    # pdb.set_trace()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
        pred_boxes = _.cuda() if args.cuda > 0 else _

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    if vis:
        im2show = np.copy(im)
    for j in range(1, len(pascal_classes)):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
    box_results = cls_dets.cpu().numpy()
    if vis:
        # cv2.imshow('test', im2show)
        # cv2.waitKey(0)
        result_path = os.path.join(args.result_dir, "OnlineDet.jpg")
        cv2.imwrite(result_path, im2show)
    return box_results


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    args = parse_args()
    load_model(args)
    app.run()
