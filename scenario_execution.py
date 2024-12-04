import ctypes as ct
import sys
import os
import cv2
import numpy as np

from ultralytics import YOLO

xosc_path = os.path.join(os.path.expanduser('~'), "sim_loop/scenarios/cut-in.xosc")
lib_path = os.path.join(os.path.expanduser('~'), "esmini")

lib_paths = {
    "linux": os.path.join(lib_path, "bin/libesminiLib.so"),
    "linux2": os.path.join(lib_path, "bin/libesminiLib.so"),
    "darwin": os.path.join(lib_path, "bin/libesminiLib.dylib"),
    "win32": os.path.join(lib_path, "esminiLib.dll"),
}
se = ct.CDLL(lib_paths[sys.platform])

# Definition of SE_ScenarioObjectState struct
class SESpeedActionStruct(ct.Structure):
    _fields_ = [
        ("id", ct.c_int),                # id of object to perform action
        ("speed", ct.c_float),
        ("transition_shape", ct.c_int),  # 0 = cubic, 1 = linear, 2 = sinusoidal, 3 = step
        ("transition_dim", ct.c_int),    # 0 = distance, 1 = rate, 2 = time
        ("transition_value", ct.c_float),
    ]

class SELaneChangeActionStruct(ct.Structure):
    _fields_ = [
        ("id", ct.c_int),                # id of object to perform action
        ("mode", ct.c_int),              # 0 = absolute, 1 = relative (own vehicle)
        ("target", ct.c_int),            # target lane id (absolute or relative)
        ("transition_shape", ct.c_int),  # 0 = cubic, 1 = linear, 2 = sinusoidal, 3 = step
        ("transition_dim", ct.c_int),    # 0 = distance, 1 = rate, 2 = time
        ("transition_value", ct.c_float),
    ]

class SELaneOffsetActionStruct(ct.Structure):
    _fields_ = [
        ("id", ct.c_int),                # id of object to perform action
        ("offset", ct.c_float),
        ("max_lateral_acc", ct.c_float),
        ("transition_shape", ct.c_int),  # 0 = cubic, 1 = linear, 2 = sinusoidal, 3 = step
    ]

class SEImage(ct.Structure):
    _fields_ = [
        ("width", ct.c_int),
        ("height", ct.c_int),
        ("pixelSize", ct.c_int),
        ("pixelFormat", ct.c_int),
        ("data", ct.POINTER(ct.c_ubyte)),
    ]

# specify some function return and argument types (needed for the floats)
se.SE_SetCameraMode.argtypes = [ct.c_int]
se.SE_SaveImagesToFile.argtypes = [ct.c_int]
se.SE_GetObjectNumberOfCollisions.argtypes = [ct.c_int]
se.SE_SaveImagesToRAM.argtypes = [ct.c_bool]
se.SE_FetchImage.argtypes = [ct.c_void_p]
se.SE_CollisionDetection.argtypes = [ct.c_bool]
se.SE_FetchImage.restype = ct.c_int

#For screenshots
#SE_SaveImagesToFile(int nrOfFrames) --> for testing purposes
#SE_SaveImageToRAM(bool state) --> also try

#Custom camera in front of vehicle, e.g. sensor mount position: 
# ./bin/esmini --window 60 60 800 400 --osc ./resources/xosc/slow-lead-vehicle.xosc --custom_camera 3,0,0.6,0,0

# specify some arguments and return types of useful functions
se.SE_StepDT.argtypes = [ct.c_float]
se.SE_GetSimulationTime.restype = ct.c_float

# initialize some structs needed for actions
lane_offset_action = SELaneOffsetActionStruct()
lane_change_action = SELaneChangeActionStruct()
speed_action = SESpeedActionStruct()
img = SEImage()

# initialize esmini with provided scenario
#se.SE_Init(sys.argv[1].encode('ascii'), 0, 1, 0, 0)

se.SE_Init(xosc_path.encode(), 0, 1, 0, 0) #3 -> no viewer, but image generated
se.SE_SetCameraMode(5) #first person view

#se.SE_SaveImagesToFile(3)
se.SE_SaveImagesToRAM(True)
se.SE_CollisionDetection(True)

# Load a model
model = YOLO("best.pt")

# Known width of the object (e.g., a car width in meters)
KNOWN_WIDTH = 2.0  # Example width in meters
# Focal length of the camera (calibrated)
FOCAL_LENGTH = 45 * 800 / 36  # Example focal length in pixels

j = 0
flag_speed_action = False
flag_braking = False
while se.SE_GetQuitFlag() == 0 and se.SE_GetSimulationTime() < 17.0:
    flag = se.SE_FetchImage(ct.byref(img))
    coll_ego = se.SE_GetObjectNumberOfCollisions(0)
    if coll_ego > 0:
        exit(-1)
    if not flag:
        total_bytes = img.pixelSize * img.width * img.height
        img_data = np.ctypeslib.as_array(img.data, shape=(total_bytes,))
        img_array = img_data.reshape((img.height, img.width, img.pixelSize, ))

        img_array = np.flip(img_array, 0) # flip y axis
        #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # change BGR to RGB
        #image_name = "output_" + str(j) + ".png"
        #cv2.imwrite(image_name, img_array)
        #results = model(f"images/{image_name}")
        #results = model(img_array)
        #results[0].save()
        results = []

        # Iterate through the results and calculate distances
        for r in results:
            for box in r.boxes:
                cls = box.cls
                conf = box.conf
                if conf >= 0.3:
                    # Calculate the width of the bounding box in pixels
                    box_width = box.xyxy[0][2] - box.xyxy[0][0]
                    ego_box_left_edge = 800/2 - box_width/2
                    object_right_edge = box.xyxy[0][2]
                    # Calculate the distance
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
                    if object_right_edge > ego_box_left_edge and distance < 30:
                        flag_braking = True

        if flag_braking and not flag_speed_action:
            print("Injecting speed action - brake")
            speed_action.id               = 0
            speed_action.speed            = 0.0
            speed_action.transition_shape = 0
            speed_action.transition_dim   = 1
            speed_action.transition_value = 7.0
            se.SE_InjectSpeedAction(ct.byref(speed_action))
            flag_speed_action = True

    se.SE_StepDT(0.1)
    j += 1