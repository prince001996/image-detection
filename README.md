# image-detection
detect objects on the uploaded image or video

this is a flask project where Rest api is used to upload an image or a video and apply object detection using yolov3 on image and yolov3 lite on videos and then teh output is shown.
the output for image is the origianl image and the table of objects and their count
the output of video is the video with bounding boxes
the frame rate of video can be cotrolled from app.py in the gen() by changing framerate
the caching by browser has been stopped so that it shows the updated images or videos

Prerquisites for the app:
python or anaconda should be installed
install the packages by checking it in the app.py on the top
Set the environment variable of python if using python without anaconda
Download yolov3.weights and yolov3-tiny.weights and put it in the project folder befor running the project
Both of them can be easily found by googling it

Steps to run the app :
open command prompt or conda prompt, change the directory to img_det
type pyhton app.py and open "http://localhost:5000/" in your browser
demo files are in the folder "Demo files"
