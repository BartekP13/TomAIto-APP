# Tomato maturity detection web application 

<div align="center">
<img src="example_picture.jpg" height="1000"/>
</div>

This is a modified code from [this](https://dev.to/andreygermanov/a-practical-introduction-to-object-detection-with-yolov8-neural-network-3n8c) article.

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8) 
implemented on [Python](https://www.python.org) that uses a model to detect level of ripeness for big and small tomatos.

## Install

* Clone this repository with this html: https://git.wmi.amu.edu.pl/s486797/tomaito_web.git
* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`

## Run

Execute:

```
python3 object_detector.py
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes of all objects detected on it.
