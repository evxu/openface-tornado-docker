#!/usr/bin/env python

# This tornado web server compares two images (from url) to judge if they are from the same source
# Detect faces and extract face features using openface https://github.com/cmusatyalab/openface
# Build webserver with tornado
# Request this server in url: http://localhots:8000/?url1=img1url&url2=img2url&score=0.48  (you can specify score or not, the default threshold socre is 0.48)
# 11/10/2016

# result:
#   True <=> two images are matched
#   False <=> two images are not matched
#   None <=> can not detect frontal faces in one of the image

import time
import os
import tornado.ioloop
import tornado.web
import tornado.options
import tornado.httpserver
import requests
import simplejson as json
import cv2
import argparse
import itertools
import StringIO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import openface

import gray2rgb from detectface

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


def extract_face(imgPath):
    # rgbImg = cv2.imread(imgPath)
    # ImgName = os.path.basename(imgPath)
    # if rgbImg is None:
    #     raise Exception("Unable to load image: {}".format(imgPath))
    # #rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    rgbImg = imgPath
    bbs = align.getAllFaceBoundingBoxes(rgbImg)
    if len(bbs) == 0:
        return 0 #'Unable to find a frontal face'

    # Draw boxes
    num_face = 0
    reps = []

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
        reps.append(rep)
    return reps

def CompareFace(reps1, reps2, score):
    if not (reps1 and reps2):
        return 'None'

    # dd = np.array([])
    for face1 in reps1:
        for face2 in reps2:
            # compute distance
            d = face1 - face2
            d = np.dot(d,d)
            if d < score:
                return 'True'
    # print 'scores:', dd
    # if np.amin(dd) < score:
    #     return 'True'
    return 'False'

class MatchFaceHandler(tornado.web.RequestHandler):

    def get(self):
        #start = time.time()
        score = self.get_argument('score', 0.48) # 0.4 can also work
        score = float(score)
        url1 = self.get_argument('url1')
        url2 = self.get_argument('url2')
        # print 'url1:'
        # print url1
        # print 'url2:'
        # print url2
        response = requests.get(url1)
        img1 = np.array(Image.open(StringIO.StringIO(response.content)))
        if len(img1.shape) == 2:
            img1 = gray2rgb(img1)

        response = requests.get(url2)
        img2 = np.array(Image.open(StringIO.StringIO(response.content)))
        if len(img2.shape) == 2:
            img1 = gray2rgb(img2)

        # do the detection
        reps1 = extract_face(img1)
        reps2 = extract_face(img2)
        # if reps1:
        #     n1 = len(reps1)
        #     print ('{} faces are detected in url1'.format(n1))
        # else:
        #     print 'None face is detected in url1'

        # if reps2:
        #     n2 = len(reps2)
        #     print ('{} faces are detected in url2'.format(n2))
        # else:
        #     print 'None face is detected in url2'

        rst = CompareFace(reps1, reps2, score)
        self.write(rst)
        #print ('processing images took {} seconds'.format(time.time()-start))
        pass


if __name__ == "__main__":
    # Setup the server
    tornado.options.parse_command_line()
    app = tornado.web.Application([(r"/", MatchFaceHandler),])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(PORT)
    tornado.ioloop.IOLoop.instance().start()

