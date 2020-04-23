import cv2
import numpy as np
from flask_caching import Cache
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify, Response, make_response
from werkzeug.utils import secure_filename
import os
from collections import Counter
import time

#Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Image detection app'
app.config["DEBUG"] = True
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
ALLOWED_EXTENSIONS_1 = {'jpg', 'jpeg', 'png'}
ALLOWED_EXTENSIONS_2 = {'mp4', '3gp'}
app.config["UPLOADS"] = os.path.join("static", "input")

#check image extensions
def allowed_file_1(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_1

#check video extensions           
def allowed_file_2(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_2
           


#stop caching in browser
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response


#object detection on video
#produces a generator object to pass it to the img tag in html using url_for
def gen(cap):
    #function to produce a frame and pass it for object detection
    def getframe(cap, sec):
        #Function to extract frames according to time
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        _, frame = cap.read()
        return frame
    
    #starting time
    sec = 0
    #skip interval for frame
    frameRate = .1
    
    starting_time = time.time()
    frame_id = 0
    #object detection on the video
    while True:
        frame = getframe(cap, sec)
        #case when the frame is returned as empty at the end of the video
        if frame is None:
            break
        
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)  
        frame_id += 1 
        sec = sec+frameRate
        height, width, channels = frame.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net2.setInput(blob)
        outs = net2.forward(output_layers2)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates27
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        #Running Non_maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
  
        #dont remove this line
        #it is used to encode the numpy array to bytes for .JPEG format and pass it to front-end using generator object 
        frame = cv2.imencode('.jpg', frame)[1].tostring()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_obj_det')
def video():
    return Response(gen(cv2.VideoCapture(os.path.join(app.config["UPLOADS"], "video.mp4"))),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        
        if file.filename == '':
            flash('Please select a file')
            return redirect(url_for("index"))
            
        if file and allowed_file_1(file.filename):

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            
            #writing the file in Uploads folder
            full_filename = os.path.join(app.config["UPLOADS"],'input.jpg' )
            cv2.imwrite(full_filename, img)
            
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            # Detecting objects in an image
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net1.setInput(blob)
            outs = net1.forward(output_layers1)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            labels = []
            for i in range(len(boxes)):
                if i in indexes:
                    labels.append(str(classes[class_ids[i]]))
                    
            labels_count = Counter(labels)
        
            return render_template("output.html", output_image = full_filename, objects = labels_count)
        
        #for videos
        if file and allowed_file_2(file.filename):
            
            file.save(os.path.join(app.config["UPLOADS"], "video.mp4"))
            
            return render_template("output2.html")
        
        flash('File not with correct extension')
        return redirect(url_for("index"))
    
#Load Yolo for image
net1 = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names1 = net1.getLayerNames()
output_layers1 = [layer_names1[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Load Yolo for video
#net2 = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net2 = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
font = cv2.FONT_HERSHEY_PLAIN
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names2 = net2.getLayerNames()
output_layers2 = [layer_names2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]


app.run()