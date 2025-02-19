import ctypes as ct
import sys
import os
import cv2
import numpy as np

from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from PIL import Image
from IPython.display import display


def extract_a_dec(osc_path_string):

    # Given string
    input_string = osc_path_string

    last_part = input_string.split('/')[-1]  # Get the last part after the last '/'

    # Split the string by underscores
    parts = last_part.split('_')

    # Extract the value between the first and second underscores
    if len(parts) > 2:
        extracted_value = parts[1]
    else:
        extracted_value = None  # Handle cases where there are not enough parts

    return extracted_value


def calculate_distance_threshold(distances, dt, j, ego_deceleration):
    """
    Calculate the distance threshold based on the given distances, time step, 
    current index, and ego deceleration.

    Args:
        distances (list): A list of distances.
        dt (float): The time step.
        j (int): The current (time) index.
        ego_deceleration (float): The ego deceleration.

    Returns:
        float: The calculated distance threshold.
    """
    if not distances or len(distances) < 2:
        return 500  # default distance threshold

    distances_without_zeros = [item for item in distances if item != 0]
    if len(distances_without_zeros) < 2:
        return 500  # default distance threshold

    not_zero_indices = [index for index, value in enumerate(distances[0:-1]) if value != 0]
    if not not_zero_indices:
        return 500  # default distance threshold

    distances_delta = distances[-1] - distances[max(not_zero_indices)]
    
    time_delta = (j - max(not_zero_indices)) * dt
    delta_v = distances_delta / time_delta

    time_delta = dt
    delta_v = (distances[-1] - distances[-2]) / dt

    if delta_v == 0 or delta_v > 40:
        return 500  # default distance threshold
    else:
        return np.square(delta_v) / (2 * ego_deceleration)


xosc_path = "/content/sim_loop/scenarios/cut-in.xosc"
lib_path = "/content/esmini"

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

# specify arguments types of esmini function
se.SE_InitWithArgs.argtypes = [ct.c_int, ct.POINTER(ct.POINTER(ct.c_char))]

# fetch command line arguments
argc = len(sys.argv)
argv = (ct.POINTER(ct.c_char) * (argc + 1))()
for i, arg in enumerate(sys.argv):
    argv[i] = ct.create_string_buffer(arg.encode('utf-8'))

# init esmini
if se.SE_InitWithArgs(argc, argv) != 0:
    exit(-1)

se.SE_SetCameraMode(5) #first person view
se.SE_SaveImagesToRAM(True)
se.SE_CollisionDetection(True)

# Load a model
model = YOLO("/content/sim_loop/gt_generation/best.pt")

# Known width of the object (e.g., a car width in meters)
KNOWN_WIDTH = 2.0  # Example width in meters
# Focal length of the camera (calibrated)
FOCAL_LENGTH = 45 * 800 / 36  # Example focal length in pixels

j = 0
delta_v = 0
dt = 0.1
ego_deceleration = 5.0
conf_level = 0.6
distances = list()
flag_speed_action = False
flag_braking = False

max_a_dec = float(extract_a_dec(sys.argv[-3]))

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
        #cv2_imshow(img_array)
        results = model(img_array)
        #results[0].save()

        # Iterate through the results and calculate distances
        for r in results:
            for box in r.boxes:
                cls = box.cls
                conf = box.conf
                if conf >= conf_level:
                    # Calculate the width of the bounding box in pixels
                    box_width = box.xyxy[0][2] - box.xyxy[0][0]
                    ego_box_left_edge = 800/2 - box_width/2
                    object_right_edge = box.xyxy[0][2]
                    # Calculate the distance
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width.item()
                    distances.append(distance)
                    distance_threshold = calculate_distance_threshold(distances, dt, j, ego_deceleration)
                    if object_right_edge > ego_box_left_edge and distance < distance_threshold:
                        flag_braking = True
                    else:
                        flag_braking = False

        if flag_braking and not flag_speed_action:
            print("Injecting speed action - brake")
            speed_action.id               = 0
            speed_action.speed            = 0.0
            speed_action.transition_shape = 0
            speed_action.transition_dim   = 1
            speed_action.transition_value = max_a_dec
            se.SE_InjectSpeedAction(ct.byref(speed_action))
            flag_speed_action = True
        
        if not flag_braking and flag_speed_action:
            print("Injecting speed action - accelerate")
            speed_action.id               = 0
            speed_action.speed            = 30.0
            speed_action.transition_shape = 0
            speed_action.transition_dim   = 1
            speed_action.transition_value = max_a_dec
            se.SE_InjectSpeedAction(ct.byref(speed_action))
            flag_speed_action = False

    se.SE_StepDT(dt)
    j += 1