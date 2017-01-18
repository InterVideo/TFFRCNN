import tensorflow as tf
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import base64
import json

from PIL import Image
from StringIO import StringIO

from flask import Flask, request

from huey.consumer import EVENT_FINISHED
from huey_queue_config import huey

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

from faster_rcnn.queue_tasks import train_exemplar_svm_on_sift_features

app = Flask(__name__)

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)


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

    detections = []

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

        detection = get_detection(im, cls, dets, CONF_THRESH)

        if detection:
            detections.append(detection)

    return detections


def get_detection(im, class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    detection = dict()

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        detection = {
            'prediction': class_name,
            'score': score,
            'x': bbox[0], 'y': bbox[1],
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1]
        }

        print bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score)

    print '{} detections with p({} | box) >= {:.1f}'.format(class_name, class_name, thresh)

    return detection


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


sess, net = None, None
SVM_MODEL = None

@app.route('/')
def welcome():
    return 'Welcome to Faster-RCNN detection service.'


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        base64_img = request.args.get('image', '').encode('utf8')
        if base64_img != '':
            detections = run_faster_rcnn_on_base64_image(sess, net, base64_img)
            return json.dumps({'detections': detections}, cls=JSONEncoder)
        else:
            return 'Image encoding error'
    elif request.method == 'POST':
        base64_img = json.loads(request.data).get('image', '')
        detections = run_faster_rcnn_on_base64_image(sess, net, base64_img)
        return json.dumps({'detections': detections}, cls=JSONEncoder)


@app.route('/train-svm', methods=['POST']):
def train_svm():
    data = json.loads(request.data)
    detections = data.get('detections', None)
    image = data.get('image', None)
    positive_crop = data.get('positive_crop', None)
    use_dense_sift = data.get('use_dense_sift', False)
    clustering = data.get('clustering', 'kmeans')

    if not (detections and image and positive_crop):
        return json.dumps({'error': 'Parameters error'})

    image = imread_from_base64(image)
    positive_crop = imread_from_base64(positive_crop)

    SVM_MODEL = train_exemplar_svm_on_sift_features(
        image, positive_crop,
        detections[0], detections[1],
        dense_sift=use_dense_sift,
        clustering=clustering
    )()

    return json.dumps({'result': 'Success'})


@app.route('/predict', methods=['POST']):
def predict():
    if SVM_MODEL is None:
        return json.dumps({
            'error': 'SVM is not initialized. First initialize it by going to /train-svm'
        })

    data = json.loads(request.data)
    test_image = data.get('image', None)

    if test_image is None:
        return json.dumps({'error': 'Image parameter is not provided or is incorrect'})

    test_image = imread_from_base64(test_image)
    result = SVM_MODEL.predict(test_image)
    return json.dumps({'result': result}, cls=JSONEncoder)


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
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    # Get a demo image from ../data/demo and encode it to base64
    # with open(os.path.join(cfg.DATA_DIR, 'demo', '001763.jpg'), 'rb') as image:
    #     base64_pineapple = base64.b64encode(image.read())

    # base64_images = [
    #     (
    #         'cat_and_dog',
    #         base64_pineapple
    #     )
    # ]

    app.run()

    # for base64_img in base64_images:
    #     print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print 'Demo for {:s}'.format(base64_img[0])
    #     print run_faster_rcnn_on_base64_image(sess, net, base64_img[1])

