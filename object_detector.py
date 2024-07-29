#from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import onnxruntime as ort
import numpy as np
import requests

#my changes
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_dir)

yolo_classes = ["b_fully_ripened",
  "b_half_ripened",
  "b_green",
  "l_fully_ripened",
  "l_half_ripened",
  "l_green"
]

#app start

app = Flask(__name__)


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    crop_type = request.form.get("cropType")
    location = request.form.get("location")
    variety = request.form.get("variety")
    boxes, orientation = detect_objects_on_image(buf.stream)
    # Do something with crop_type, location, and variety here (e.g., store in database)

    return jsonify(boxes)


def detect_objects_on_image(buf):
    input, img_width, img_height = prepare_input(buf)
    output = run_model(input)
    orientation = get_orientation(buf)
    processed_output = process_output(output, img_width, img_height, orientation)
    return processed_output, orientation

def prepare_input(buf):
    img = Image.open(buf)
    img_width, img_height = img.size
    img = img.resize((640, 640))
    img = img.convert("RGB")
    input = np.array(img)
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640) / 255.0
    return input.astype(np.float32), img_width, img_height

def run_model(input):
    model = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
    outputs = model.run(["output0"], {"images":input})
    return outputs[0]

def process_output(output, img_width, img_height, orientation):
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue

        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height

        boxes.append([x1, y1, x2, y2, label, prob])

    # Adjust boxes based on orientation
    adjusted_boxes = adjust_boxes_for_orientation(boxes, orientation, img_width, img_height)

    # Sort and apply non-max suppression as before
    adjusted_boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(adjusted_boxes) > 0:
        result.append(adjusted_boxes[0])
        adjusted_boxes = [box for box in adjusted_boxes if iou(box, adjusted_boxes[0]) < 0.7]

    return result


def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)

def get_orientation(image_path):
    with Image.open(image_path) as img:
        if hasattr(img, '_getexif'):
            exif_data = img._getexif()
            if exif_data is not None:
                return exif_data.get(274, 1)  # Default to normal orientation
    return 1  # Default orientation if no EXIF data

def adjust_boxes_for_orientation(boxes, orientation, img_width, img_height):
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2, label, prob = box

        # Apply transformations based on orientation
        if orientation == 3:  # 180 degrees
            x1, y1, x2, y2 = img_width - x2, img_height - y2, img_width - x1, img_height - y1
        elif orientation == 6:  # 270 degrees (or -90 degrees)
            x1, y1, x2, y2 = img_height - y2, x1, img_height - y1, x2
        elif orientation == 8:  # 90 degrees
            x1, y1, x2, y2 = y1, img_width - x2, y2, img_width - x1

        adjusted_boxes.append([x1, y1, x2, y2, label, prob])

    return adjusted_boxes


""" def detect_objects_on_image(buf):
    """ 
""""
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
"""""
    model = YOLO("best.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output
 """
serve(app, host='0.0.0.0', port=8080)