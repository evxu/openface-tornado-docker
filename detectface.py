#!/usr/bin/env python

# process image with openface
# return bounding boxes and face features

import os
import tornado.ioloop
import tornado.web
import requests
import simplejson as json
import cv2
import argparse
import itertools
import StringIO
import matplotlib.pyplot as plt

import numpy as np
import openface
from PIL import Image

PORT=8000
np.set_printoptions(precision=2)

#fileDir = os.path.dirname(os.path.realpath(__file__))
fileDir = './root/openface/demos'
modelDir = os.path.join(fileDir, '..', 'models') 
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

# parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
# parser.add_argument('output', type=str, help='Output path')
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
# parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# if args.verbose:
#     print("Argument parsing and loading libraries took {} seconds.".format(
#         time.time() - start))

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
# if args.verbose:
#     print("Loading the dlib and OpenFace models took {} seconds.".format(
#         time.time() - start))


def drawrectagle(rgbImg):
    bbs = align.getAllFaceBoundingBoxes(rgbImg)
    if len(bbs) == 0:
        return 0 #'Unable to find a frontal face'

    num_face = 0
    boundings = []

    # draw boxes and extract features
    for bb in bbs:
        # start = time.time()
        landmarks = align.findLandmarks(rgbImg, bb)
        alignedFace = align.align(args.imgDim, rgbImg, bb,
                                    landmarks = landmarks,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            continue
        num_face += 1
        rep = net.forward(alignedFace)
        boundings.append({'box': [bb.left(), bb.top(), bb.width(), bb.height()], 'feature':rep.tolist()})
        # bl = (bb.left(), bb.bottom())
        # tr = (bb.right(), bb.top())
        # cv2.rectangle(annotatedImg,  bl, tr, color=(153, 255, 204),
        #               thickness=3)
        # for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
        #     cv2.circle(annotatedImg, center=landmarks[p], radius=3,
        #                color=(102, 204, 255), thickness=-1)
    return boundings


class DetectFaceHandler(tornado.web.RequestHandler):
    def get(self):
        # get image from url
        url = self.get_argument('url')
        response = requests.get(url)
        rgbImg = np.array(Image.open(StringIO.StringIO(response.content)))
        boundings = drawrectagle(rgbImg)
        results = {'results': boundings, 'shape':rgbImg.shape}
        self.write(json.dumps(results))
        pass

    def post (self):
        # 'image' below corresponds to -F "image=@..." in curl command line
        fileinfo = self.request.files['image'][0]
        # this is the binary content of an image
        # to turn this into an numpy 
        file_bytes = np.asarray(bytearray(fileinfo.body), dtype=np.uint8)
        bgrImg = cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_COLOR)
        # color image loaded by OpenCV is in BGR mode, should convert it to RGB mode.
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        # do the detection
        boundings = drawrectagle(rgbImg)
        # assume you have results in the following dict
        # results = {'results': [
        #                   # x, y, w, h
        #             {' box':[1, 2, 3, 4], 'feature':[0,0,0,0,0]},
        #             {'box':[2, 2, 3, 4], 'feature':[0,0,0,0,0]},
        #         ],
        #         'shape': image.shape # check if image is correctly loaded

        #         };
        results = {'results': boundings, 'shape':rgbImg.shape}
        self.write(json.dumps(results))
        pass


application = tornado.web.Application([
    (r"/", DetectFaceHandler),
])

if __name__ == "__main__":
    # Setup the server
    application.listen(PORT)
    tornado.ioloop.IOLoop.instance().start()

