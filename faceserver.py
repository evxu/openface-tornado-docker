from detectface import DetectFaceHandler
from matchface import MatchFaceHandler
import tornado.ioloop
import tornado.web
import tornado.options

# To detect face and extract face features by post file:
# 	curl -F image=@xx.jpg http://localhost:8000/detect/

# To detect face and extract face features by url:
#	curl 'http://localhost:8000/?url=imageurl'

# To match face by url:
#	curl 'http://locahost:8000/match/?url1=url_of_image1&url2=url_of_image2'
PORT = 8000

if __name__ == "__main__":
    # Setup the server
    tornado.options.parse_command_line()
    app = tornado.web.Application(
    	handlers=[
    	(r"/match/", MatchFaceHandler),
    	(r"/detect/", DetectFaceHandler)
    	]
    )
    app.listen(PORT)
    tornado.ioloop.IOLoop.instance().start()
