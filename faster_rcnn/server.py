import tensorflow as tf
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import base64

from PIL import Image
from StringIO import StringIO

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def imread_from_base64(base64_str):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_str))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def run_faster_rcnn_on_base64_image(sess, net, base64_img):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = imread_from_base64(base64_img)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]


        # TODO
        # This prints out all detections for all classes.
        # Look at vis_detections function in ./demo.py to
        # get the desired behavior.
        print '======= DETECTION RESULT =========='
        print 'IMAGE:', im
        print '-----------------------------------'
        print 'CLS:', cls
        print '-----------------------------------'
        print 'DETS:', dets
        print '==================================='


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ' or not os.path.exists(args.model):
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print (' done.')

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _ = im_detect(sess, net, im)

    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    # Get a demo image from ../data/demo and encode it to base64
    with open(os.path.join(cfg.DATA_DIR, 'demo', 'pineapple.jpg'), 'rb') as image:
        base64_pineapple = base64.b64encode(image.read())

    base64_images = [
        (
            'pineapple',
            base64_pineapple
        )
    ]

    for base64_img in base64_images:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(base64_img[0])
        run_faster_rcnn_on_base64_image(sess, net, base64_img[1])
