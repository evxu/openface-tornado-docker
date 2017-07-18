# openface-tornado
Source code for openface-tornado docker image: https://hub.docker.com/r/evxu/openface-tornado/ 
---
#### Description
A face detection API based on [OpenFace](https://cmusatyalab.github.io/openface/) and tornado web server.  It can help you detect faces, extract face features, draw bounding boxes on the detected faces, and compare faces between two images.

---

#### Installation
```
$ docker pull evxu/openface-tornado:latest
```

---

#### Run the server
Run the docker container by:
```
$ docker run -i -t -p 9000:8000 evxu/openface-tornado --name pinkring /bin/bash
$ python ./oft/demo/faceserver.py
```
Or run  the default demo in background:
```
$ docker run -p 9000:8000 -d --name pinkring evxu/openface-tornado
```
Now, you have built a container named 'pinkring' within our API running.
To mount your own file, use `-v` option.
 
Let's test your installation on your host:

**1.  Test the face matching server:**
In command:
```
curl 'http://localhost:9000/matchface/?url1=http://images.idspicturedesk.com/images/161/lowres/abexs/abexs411.jpg&url2=http://images.idspicturedesk.com/images/161/lowres/abexs/abexs411.jpg'
```
Or visit the hTTP address in your browser.
It should return you `{"distance": 0.0}` in JSON data format, because url1 and url2 are the same image.

**2. Test the face detection server with url:**
In command:
```
curl 'http://localhost:9000/detectface/?url=http://images.idspicturedesk.com/images/161/lowres/abexs/abexs411.jpg'
```
Or visit the hTTP address in your browser.
It should return you the data in JSON format starting with `{"shape": [350, 600, 3], "results": [{"box": [380, 109, 130, 130], "feature": [-0.0076855164952576, ...`
`response.shape` returns the the shape of the image.
 `response.results` gives you the information of locations and face features of detected face,
 where `response.results[0].box` represents the bounding box of first detected face in the format of [left, top, width, height], and `response.results[0].feature` is the feature in format of `list(128,)` extracted from the first detected face.

**3. Test the face detection server with image file:**
in command:
```
curl -F image=@xxxx.jpg http://localhost:9000/detect/
```
It will return data in the same format as that when you post image in URL.

**4. Show the detected face in bounding boxes**

Open your browser and visit:
```
http://localhost:9000/boxface/?url=http://images.idspicturedesk.com/images/161/lowres/abexs/abexs411.jpg
```
It will display the image with bounding boxes on detected faces.

---
#### Usage
To request face matching server, request the URL on the host:
`curl 'http://localhost:9000/matchface/?url1=[url of first image]&url2=[url of second image]'`

To request face detection server:
post image with URL:
`curl 'http://localhost:9000/detectface/?url=[url of image]'`
post image with a image file:
`curl -F image=@[image path] http://localhost:9000/detect/'`

To draw bounding boxes on faces:
The URL is `http://localhost:9000/?boxface/?url=[url of image]`
If you are running this API on a remote server. Map the port of your local machine to the remote server by `ssh -L 9000:localhost:9000 [address of remote server]` . Then you should be able to check the results with you local browser.


##### in Python
To request face detection server:
```Python
import request
# if use url
url = 'your image url'
resp = requests.get('http://localhost:9000/detectface/ulr='+url) 
# if use file
imgpath='path of your image file'
resp = requests.post('http://localhost:9000/detectface/', files = {'image': open(imgpath, 'r')})
features = []
boxes = []
rst = resp.json()['results']
if rst: # if there are faces detected
      features += [b['feature']]
      boxes += [b['box']]
```
