import io
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import rospy
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Int8
from publisher import  PedestrianLeftPublisher, PedestrianRightPublisher, VehicleLeftPublisher, VehicleRightPublisher, FrontImagePublisher, BackImagePublisher

MIN_NUM_FRAMES = 10


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
    if mode == 'ped':
        return relative_displacement < 0.1, relative_displacement # and area_change == 0
    elif mode == 'veh':
        if relative_displacement < 0.2:
            if False not in movement:
                return 'ignore', relative_displacement
            else:
                return True, relative_displacement
        else:
            return False, relative_displacement


class DetectionRobot:
    def __init__(self):
        rospy.init_node('detect_universal', anonymous=True)

        self.front_img = None
        self.back_img = None
        self.position_data = None
        self.rotation_data = None
        self.current_pos = None
        self.current_vel = None
        self.centers_old = {}
        self.movement_dict = {}
 
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load YOLO model
        self.model = YOLO('yolo11x.pt').to(self.device)

        # Publishers
        self.publishers = [PedestrianLeftPublisher(), PedestrianRightPublisher(), VehicleLeftPublisher(), VehicleRightPublisher()]

        # Initialize image publishers
        self.front_pub = FrontImagePublisher()
        self.back_pub = BackImagePublisher()
 
        #External topics subscription
        rospy.Subscriber('/head_front_camera/color/image_raw/compressed', CompressedImage, self.front_cam_callback)
        rospy.Subscriber('/cam_new_back/color/image_raw/compressed', CompressedImage, self.back_cam_callback)
        rospy.Subscriber('/position', Int8, self.position_callback)
        rospy.Subscriber('/head_controller/state', JointState, self.rotation_callback)

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

    def rotation_callback(self, msg):
        self.current_pos = msg.actual.positions[0]
        self.current_vel = msg.actual.velocities[0]

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

    def rotation_callback(self, msg):
        self.current_pos = msg.actual.positions[0]
        self.current_vel = msg.actual.velocities[0]


    def detect_universal_fb(self, mode: str) -> int:
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
        centers_old = {}
        movement_dict = {}
        lastKey = ''

        frame = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in [self.front_img, self.back_img]]
        results = self.model.predict(frame, conf=conf_level, classes=[0] if mode == "ped" else [2,3,4,6,8], device=self.device, verbose=False)
        
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes = boxes[boxes[:,0].argsort()]  # Sort boxes by x1 (leftmost) coordinate
            conf = result.boxes.conf.cpu().numpy()
            
            # Draw detections on frame
            frame_with_boxes = frame[i].copy()
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Publish annotated frames
            if i == 0:
                self.front_pub.publish(frame_with_boxes)
            else:
                self.back_pub.publish(frame_with_boxes)
            
            frame_movement = {}
            is_first = True
            # Process detections
            for ix, (box, confidence) in enumerate(zip(boxes, conf)):
                xmin, ymin, xmax, ymax = map(int, box)
                width = xmax - xmin
                height = ymax - ymin
                
                center_x = int((xmax + xmin) / 2)
                center_y = int((ymax + ymin) / 2)
                
                # Update tracking with both position and size
                centers_old, id_obj, is_new, lastKey = update_tracking(
                    centers_old, 
                    (center_x, center_y),  # position
                    (width, height),       # size
                    thr_centers, 
                    lastKey, 
                    i, 
                    frame_max
                )

                if id_obj not in movement_dict:
                    movement_dict[id_obj] = []

                stationary, displacement = is_stationary(centers_old[id_obj], movement_dict[id_obj][MIN_NUM_FRAMES:], mode='veh')
                movement_dict[id_obj].append(stationary)


                if mode == 'ped':
                    frame_movement[id_obj] = stationary
                else:
                    if is_first and stationary == True:
                        frame_movement[id_obj] = True
                        is_first = False
                    else:
                        is_true = False if stationary != 'ignore' else is_true
                        frame_movement[id_obj] = False
                    
                    
            if mode == "ped":
                if any(frame_movement.values()):
                    self.publishers[i].publish(True)
                else:
                    self.publishers[i].publish(False)
            else:
                if any(frame_movement.values()):
                    self.publishers[i+2].publish(True)
                else:
                    self.publishers[i+2].publish(False)
        
    
    def run_detection(self):
 
        rate = rospy.Rate(20)
 
        while not rospy.is_shutdown():
            if self.position_data == 0:
                mode = 'ped'
            elif self.position_data == 1:
                mode = 'veh'
            else:
                mode = 'sleep'


            if mode in ['ped', 'veh'] and self.front_img is not None and self.back_img is not None and self.current_vel is not None and abs(self.current_vel) < 1e-2:
                self.detect_universal_fb(mode=mode)

            rate.sleep()

if __name__ == '__main__':
    try:
        robot = DetectionRobot()
        rospy.loginfo("Init Robot Stat")
        robot.run_detection()
 
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection Robot node terminated.")