#!/usr/bin/env python
# read image path from input.txt
# read corresponding names in iamge from input_names.txt, where images are in the same order as input.txt

# convert image to smaller size, store in image/iptc_img1
# post image to http (demo3.py)
# write result to result.txt
# draw rectangle with opencv based on results from http, restore image in public_html/iptc_img/
# write two html pages

# or read image path from stdin:
# run test_tornado2.py by command:
# find folder/* -type f |./test_tornado2.py
# for img in stdin:
# 	img = img.strip()

import commands
import cv2
import numpy as np
import subprocess
import os
import requests

convert_img = False
input_file = 'input.txt'
name_file = 'input_names.txt'
fn = open(name_file, 'r')
frst = open('result.txt', 'w')
fd = open('../public_html/tornado_detected.html', 'w')    
fu = open('../public_html/tornado_undetected.html','w')

def drawrectagle(imgPath, boxes):
	rgbImg = cv2.imread(imgPath)
	annotatedImg = np.copy(rgbImg)
	for bb in boxes:
		bb = bb['box'] #[bb.left(), bb.top(), bb.width(), bb.height()]
		vertex1 = (bb[0], bb[1])
		vertex2 = (bb[0] + bb[2], bb[1] + bb[3])
		cv2.rectangle(annotatedImg, vertex1, vertex2, color=(153,255,284), thickness=3)
	return annotatedImg

converted_path = '/home/wang/image/iptc_img1/'
if __name__ == "__main__":
	for img in open(input_file, 'r'):
		img = img.strip()
		if convert_img: # use raw image
			ImgName = os.path.basename(img) + '.jpg'
			# convert image
			subprocess.call('convert '+img + ' -resize 800 ' + converted_path + ImgName, shell=True)

			img = converted_path+ImgName
		# use converted image
		else:
			ImgName = ImgName = os.path.basename(img)

		resp = requests.post('http://localhost:9000/', files={'image': open(img,'r')})
		result = str(resp.json())
		frst.write(img + '\t' + result+'\n')  # write result to result.txt
		# draw bounding boxes
		# if there is not detected face, results is 0

		# get names in image
		names = fn.readline()		
		names = names.strip()
		names = names.split('\t')[1:]
		names = '\t'.join(names)

		boxes = resp.json()['results']
		if boxes:
			num_face = len(boxes)
			annotatedImg = drawrectagle(img, boxes)
			# save image with boundinb boxes to public_html
			NewImgName = 'Annotated' + '_' + str(num_face) + '_'+ ImgName
			cv2.imwrite('/home/wang/public_html/iptc_img/' + NewImgName, annotatedImg)
			# write html
			fd.write('<a href="./iptc_img/{0}"><img src="./iptc_img/{0}" width="400"></img></a><pre>{1}</pre>'.format(NewImgName, names))
			print ('Detected {0} faces in {1}'.format(num_face, ImgName))
		else:
			NewImgName = 'NoFrontalFace_' + ImgName
			subprocess.call('cp ' + img + ' /home/wang/public_html/iptc_img/' + NewImgName, shell = True)
			//# write html
			fu.write('<a href="./iptc_img/{0}"><img src="./iptc_img/{0}" width="400"></img></a><pre>{1}</pre>'.format(NewImgName, names)) 
			print ('Did not find frontal face in {0}'.format(ImgName))
	frst.close()
	fn.close()
	fd.close()
	fu.close()