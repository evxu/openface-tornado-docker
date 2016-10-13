FROM bamos/openface
MAINTAINER Xueqi Wang

RUN mkdir -p /oft/src
RUN mkdir -p /oft/demo

# Add and install Python modules
ADD ./requirements.txt /oft/src/requirements.txt
RUN cd oft/src; pip install -r requirements.txt

ADD ./detectface.py /oft/demo/
ADD ./matchface.py /oft/demo/
ADD ./faceserver.py /oft/demo/

EXPOSE 8000

CMD python /oft/demo/faceserver.py