from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")
#model = YOLO("yolo11s.pt")

# Perform object detection on an image
results = model("images/output_74.png")
results[0].show()