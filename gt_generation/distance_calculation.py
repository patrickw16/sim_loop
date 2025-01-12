#F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth (pixel)
#F(pixels) = F(mm) * ImageWidth(pixel) / SensorWidth(mm)
#Sensor: 36x24mm
#Value(mm): 20,08054

#Image: 800x400 Pixel

#Bounding box dimensions: width=2m, length=5.04m, height=1.5m

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Load an image
image_path = r"/home/patrick_w/sim_loop/_old/figures/output_81.png"
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Known width of the object (e.g., a car width in meters)
KNOWN_WIDTH = 2.0  # Example width in meters

# Focal length of the camera (calibrated)
FOCAL_LENGTH = 45 * 800 / 36  # Example focal length in pixels

# Iterate through the results and calculate distances
for r in results:
    for box in r.boxes:
        cls = box.cls
        conf = box.conf
        if conf >= 0.3:
            # Calculate the width of the bounding box in pixels
            box_width = box.xyxy[0][2] - box.xyxy[0][0]
            #Lateral distance:
            #ego camera is always at 400 (middle of 800, width of image)
            #ego is same width as object -> use same box_width
            #determine "begin" of ego bounding box -> 400 - box_width/2
            #all in pixels: if bounding boxes (right edge of object and left edge of ego) overlap (+ distance smaller than x (determined with point mass model for braking)) then brake
            ego_box_left_edge = 800/2 - box_width/2
            object_right_edge = box.xyxy[0][2]
            print(object_right_edge, ego_box_left_edge)
            # Calculate the distance
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
            print(f"Object: {model.names[int(cls)]}, Distance: {distance:.2f} meters")

