import io
import cv2
import time
from ultralytics import YOLO
import numpy as np
import torch
import rospy
import openvino as ov
from pathlib import Path
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Int8
from publisher import  PedestrianLeftPublisher, PedestrianRightPublisher, VehicleLeftPublisher, VehicleRightPublisher, FrontImagePublisher, BackImagePublisher
from PIL import Image

MIN_NUM_FRAMES = 10
MIN_NUM_FRAMES_VEH = 3


def filter_tracks(centers: dict, patience: int) -> dict:
    """Function to filter track history"""
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])
    return filter_dict


def update_tracking(centers_old: dict, obj_center: tuple, obj_size: tuple, 
                   thr_centers: int, lastKey: str, frame: int, frame_max: int) -> tuple:
    """Function to update track of objects"""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) 
               for k, center in centers_old.items()]
    lastpos = [(i[0], i[2][0]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    
    # Calculating distance from existing centers points
    previous_pos = [(k,obj_center) for k,centers in lastpos 
                   if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
    
    # if distance less than a threshold, it will update its positions
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = (obj_center, obj_size)
    # Else a new ID will be set to the given object
    else:
        if lastKey:
            last = lastKey.split('D')[1]
            id_obj = 'ID' + str(int(last)+1)
        else:
            id_obj = 'ID0'
        is_new = 1
        centers_old[id_obj] = {frame: (obj_center, obj_size)}
        lastKey = list(centers_old.keys())[-1]
    return centers_old, id_obj, is_new, lastKey

def is_stationary(positions: dict, movement: list, mode: str = 'ped', min_displacement: float = 0.4, min_positions: int = MIN_NUM_FRAMES) -> bool:
    """
    Determine if an object is stationary based on:
    1. Position displacement relative to object size
    2. Changes in object size/area
    
    Args:
        positions: Dictionary of frame numbers and positions and sizes {frame: ((x,y), (width,height))}
        min_displacement: Minimum relative displacement to consider as movement
        min_positions: Minimum number of positions needed to make a decision
        
    Returns:
        Boolean indicating if object is stationary
    """
    # Need minimum number of positions to determine if stationary
    # print(f'[INFO] Mode: {mode}')
    if len(positions) < min_positions:
        return False if mode == 'ped' else 'ignore', 0.0   # Not enough data to determine if stationary
        
    # Get last few positions and sizes
    recent_data = list(positions.values())[-min_positions:]
    
    total_displacement = 0
    for i in range(len(recent_data)-1):
        pos1 = np.array(recent_data[i][0])    # Current position
        pos2 = np.array(recent_data[i+1][0])  # Next position
        displacement = np.linalg.norm(pos2 - pos1)
        total_displacement += displacement
    
    avg_displacement = total_displacement / (len(recent_data) - 1)
    
    
    # Calculate average object size
    avg_width = np.mean([data[1][0] for data in recent_data])
    avg_height = np.mean([data[1][1] for data in recent_data])
    avg_area = avg_width * avg_height
    
    # Calculate size change
    start_area = recent_data[0][1][0] * recent_data[0][1][1]
    end_area = recent_data[-1][1][0] * recent_data[-1][1][1]
    area_change = abs(end_area - start_area) / start_area
    
    # Weight displacement by object size (larger objects need more absolute displacement)
    relative_displacement = avg_displacement * (1/(avg_area/1000))
    
    # Object is stationary if both:
    # 1. Relative displacement is insignificant
    # 2. Size hasn't changed significantly
    print(mode, relative_displacement)
    if mode == 'ped':
        return relative_displacement < 0.1, relative_displacement # and area_change == 0
    elif mode == 'veh':
        if relative_displacement < 0.2:
            return True, relative_displacement
        else:
            return False, relative_displacement


class DetectionRobot:
    def __init__(self):
        rospy.init_node('detect_universal')

        self.front_img = None
        self.back_img = None
        self.position_data = None
        self.rotation_data = None
        self.current_pos = None
        self.current_vel = None
        self.centers_old = {}
        self.movement_dict = {}
        self.frame = 0

        self.previous_state = None
        self.ref_frame_front = None
        self.ref_frame_back = None
 
        # Set device
        self.device = 'CPU' #'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load YOLO model
        core = ov.Core()
        # self.model = YOLO('yolo11m.pt').to(self.device)
        DET_MODEL_NAME = 'yolo11m'
        det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
        det_ov_model = core.read_model(det_model_path)

        # Compile the model
        det_compiled_model = core.compile_model(det_ov_model, self.device, {})
        
        # Initialize YOLO model
        self.model = YOLO(det_model_path.parent, task="detect")
        # self.model = YOLO('yolo11x.pt')

        if self.model.predictor is None:
            custom = {"conf": 0.25, "batch": 2, "save": False, "mode": "predict"}  # method defaults
            args = {**self.model.overrides, **custom}
            self.model.predictor = self.model._smart_load("predictor")(overrides=args, _callbacks=self.model.callbacks)
            self.model.predictor.setup_model(model=self.model.model)
        
        # Assign the compiled OpenVINO model to the predictor
        self.model.predictor.model.ov_compiled_model = det_compiled_model

        # Publishers
        self.publishers = [PedestrianLeftPublisher(), PedestrianRightPublisher(), VehicleLeftPublisher(), VehicleRightPublisher()]

        # Initialize image publishers
        self.front_pub = FrontImagePublisher()
        self.back_pub = BackImagePublisher()
 
        #External topics subscription
        rospy.Subscriber('/head_front_camera/color/image_raw/compressed', CompressedImage, self.front_cam_callback)
        rospy.Subscriber('/cam_new_back/color/image_raw/compressed', CompressedImage, self.back_cam_callback)
        rospy.Subscriber('/current_state', Int8, self.position_callback)

    def front_cam_callback(self, msg):
        try:
            image = PILImage.open(io.BytesIO(msg.data))
            self.front_img = np.array(image)
        except ValueError:
            print(f"Error processing message from front cam")

    def back_cam_callback(self, msg):
        try:
            image = PILImage.open(io.BytesIO(msg.data))
            back_img = np.array(image)
            
            h, w = back_img.shape[:2]
            back_img = cv2.resize(back_img, (w//2, h//2))
            back_img = np.pad(back_img, ((60,60), (0,0), (0,0)), mode='constant')
            self.back_img = back_img
        except ValueError:
            print(f"Error processing message from back cam")

    def position_callback(self, msg):
        self.position_data = msg.data


    def compute_diff_mask(self, frame: np.ndarray, direction: int) -> np.ndarray:
        # Convert to float32 for calculations
        img1 = frame.astype(np.float32)
        if direction == 0:
            img2 = self.ref_frame_front.astype(np.float32)
        else:
            img2 = self.ref_frame_back.astype(np.float32)
        
        # Calculate absolute difference for each channel
        diff = np.abs(img1 - img2)
        
        # Method 1: Sum differences across channels
        sum_diff = np.sum(diff, axis=2)
        # Normalize to 0-255 range
        sum_diff = (sum_diff * 255.0 / sum_diff.max()).astype(np.uint8)

         
        kernel = np.ones((5,5), np.uint8)
        sum_mask = cv2.morphologyEx(sum_diff, cv2.MORPH_OPEN, kernel)
        sum_mask = cv2.morphologyEx(sum_mask, cv2.MORPH_CLOSE, kernel)
    
        return sum_mask
    
    def filter_detections(self, frame: np.ndarray, boxes: np.ndarray, conf: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
        diff_mask = self.compute_diff_mask(frame, direction)
        diff_mask = np.where(diff_mask > 20, 255, 0)
        # Calculate percentage of changed pixels in each box
        filtered_boxes = []
        filtered_conf = []
        for box, confidence in zip(boxes, conf):
            xmin, ymin, xmax, ymax = map(int, box)
            # Get the region of the diff mask corresponding to the box
            box_mask = diff_mask[ymin:ymax, xmin:xmax]
            # Calculate percentage of changed pixels (non-zero values)
            changed_pixels = np.count_nonzero(box_mask)
            total_pixels = box_mask.size
            change_percentage = changed_pixels / total_pixels * 100
            
            # Keep box if more than 50% pixels changed
            if change_percentage >= 40:
                filtered_boxes.append(box)
                filtered_conf.append(confidence)

        return np.array(filtered_boxes), np.array(filtered_conf)
    

    def detect_universal_fb(self, mode: str) -> int:
        # print(f'[INFO] Detecting universal FB')
        # Configurations
        conf_level = 0.25
        thr_centers = 30
        frame_max = 10
        
        # Get video properties
        height = 480 #int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 640 #int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # fps = video.get(cv2.CAP_PROP_FPS)
        fps = 30
        
        # Initialize tracking variables
        lastKey = ''

        frame = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in [self.front_img, self.back_img]]
        results = self.model.predict(frame, conf=conf_level, classes=[0] if mode == "ped" else [1,2,3,5,7], device=self.device, verbose=False)
        

        front = None
        back = None

        for direction, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes = boxes[boxes[:,0].argsort()]  # Sort boxes by x1 (leftmost) coordinate
            conf = result.boxes.conf.cpu().numpy()
            
            frame_movement = {}
            is_first = True
            frame_with_boxes = frame[direction].copy()

            if mode == 'veh':
                boxes, conf = self.filter_detections(frame[direction], boxes, conf, direction)

            # Process detections
            for ix, (box, confidence) in enumerate(zip(boxes, conf)):
                xmin, ymin, xmax, ymax = map(int, box)
                width = xmax - xmin
                height = ymax - ymin
                
                center_x = int((xmax + xmin) / 2)
                center_y = int((ymax + ymin) / 2)
                
                # Update tracking with both position and size
                # print(self.centers_old[id_obj].keys()[-1])
                self.centers_old, id_obj, is_new, lastKey = update_tracking(
                    self.centers_old, 
                    (center_x, center_y),  # position
                    (width, height),       # size
                    thr_centers, 
                    lastKey, 
                    self.frame, 
                    frame_max
                )

                if id_obj not in self.movement_dict:
                    self.movement_dict[id_obj] = []

                if mode == 'ped':
                    stationary, displacement = is_stationary(self.centers_old[id_obj], self.movement_dict[id_obj], mode=mode)
                    self.movement_dict[id_obj].append(stationary)
                    frame_movement[id_obj] = stationary
                else:
                    stationary, velocity = is_stationary(self.centers_old[id_obj], self.movement_dict[id_obj])
                    frame_movement[id_obj] = stationary
                    self.movement_dict[id_obj].append(stationary)

                    if is_first and stationary == True:
                        frame_movement[id_obj] = True
                        is_first = False
                    else:
                        is_first = False
                        frame_movement[id_obj] = False
                # Add velocity text to image
                if mode != 'ped':
                    velocity_text = f"ID: {id_obj}"
                    cv2.putText(frame_with_boxes, velocity_text, (xmin, ymin-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0) if frame_movement[id_obj] else (0, 0, 255))

                # Publish annotated frames
            print(f'[INFO] Direction value: {direction}')
            if direction == 0:
                self.front_pub.publish(frame_with_boxes)
                front = frame_with_boxes
                
            else:
                self.back_pub.publish(frame_with_boxes)
                back = frame_with_boxes

                    
            print(f'[INFO] Frame movement: {frame_movement}')
            self.frame += 1
            if mode == "ped":
                if any(frame_movement.values()):
                    self.publishers[direction].publish(True)
                else:
                    self.publishers[direction].publish(False)
            else:
                if any(frame_movement.values()) or len(boxes) == 0:
                    self.publishers[direction+2].publish(True)
                else:
                    self.publishers[direction+2].publish(False)
        
        return front, back
    
    def run_detection(self):
        rate = rospy.Rate(10)
        front_vid = []
        back_vid = []

        while not rospy.is_shutdown():
            # print(f'[INFO] Current data: {self.position_data}\n{self.current_vel}\n{type(self.front_img)}\n{type(self.back_img)}')
            if self.previous_state is None or self.previous_state != self.position_data:
                self.previous_state = self.position_data
                self.frame = 0

            if self.position_data == 0:
                    mode = 'ped'
            elif self.position_data == 1:
                mode = 'veh'
            elif self.position_data == 2:
                print(f'[INFO] Saving videos')
                print(f'[INFO] Front vid: {len(front_vid)}')
                print(f'[INFO] Back vid: {len(back_vid)}')

                
                # Save accumulated frames as videos
                if len(front_vid) > 0:
                    front_vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame 
                               for frame in front_vid]
                    front_vid = np.array(front_vid)
                    print(f'[INFO] Front vid: {front_vid.shape}')
                    h, w = front_vid[0].shape[:2]
                    front_writer = cv2.VideoWriter('tmp/front_video.mp4', 
                                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                                 7, (w,h))
                    for num, frame in enumerate(front_vid):
                        # cv2.imwrite(f'tmp/frames/front_frame_{num:04d}.jpg', frame)
                        front_writer.write(frame)
                    front_writer.release()
                    front_vid = []
                
                if len(back_vid) > 0:
                    back_vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame 
                               for frame in back_vid]
                    back_vid = np.array(back_vid)
                    print(f'[INFO] Back vid: {back_vid.shape}')
                    h, w = back_vid[0].shape[:2]
                    back_writer = cv2.VideoWriter('tmp/back_video.mp4',
                                                cv2.VideoWriter_fourcc(*'mp4v'),
                                                7, (w,h))
                    for frame in back_vid:
                        back_writer.write(frame)
                    back_writer.release()
                    back_vid = []
                
                mode = 'save'
            else:
                mode = 'sleep'

            if self.frame == 0:
                if self.front_img is not None:
                    self.ref_frame_front = cv2.cvtColor(self.front_img, cv2.COLOR_RGB2BGR)
                if self.back_img is not None:
                    self.ref_frame_back = cv2.cvtColor(self.back_img, cv2.COLOR_RGB2BGR)

            if mode in ['ped', 'veh'] and self.front_img is not None and self.back_img is not None: # and self.current_vel is not None and abs(self.current_vel) < 1e-2:
                front, back = self.detect_universal_fb(mode=mode)
                if front is not None and isinstance(front, np.ndarray):
                    front_vid.append(front)

                if back is not None and isinstance(back, np.ndarray):
                    back_vid.append(back)

            rate.sleep()

if __name__ == '__main__':
    try:
        print(f'[INFO] Starting Detection Robot')
        robot = DetectionRobot()
        print(f'[INFO] Detection Robot initialized')
        rospy.loginfo("Init Robot Stat")
        robot.run_detection()
 
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection Robot node terminated.")