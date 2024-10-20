import sys
import numpy as np
from numpy import linalg
import cmath
import time
from ultralytics import YOLO
from math import sqrt,pi,cos,sin, atan2, acos, asin
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from controller import Robot, VacuumGripper ,  Motor , Camera, RangeFinder
    
    
    
# create the Robot instance.
robot = Robot()

##IK

global mat
mat = np.matrix

pos_new = np.zeros((1,6))

# ****** Coefficients ******

global d1, a2, a3, a7, d4, d5, d6
d1 = 0.1625
a2 = -0.425
a3 = -0.39225
a7 = 0.075
d4 = 0.133
d5 = 0.0997
d6 = 0.101

global d, alph

d = mat([0.1625, 0, 0, 0.133, 0.0997, 0.101])
a = mat([0, -0.425, -0.39225, 0, 0, 0])
alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0])

# ************************************************** FORWARD KINEMATICS

def AH(n, th, c):
    a = np.array([0, -0.425, -0.39225, 0, 0, 0])
    T_a = mat(np.identity(4), copy=False,dtype = np.float32)
    T_a[0, 3] = a[n-1]
    T_d = mat(np.identity(4), copy=False)
    T_d[2, 3] = d[0, n-1]

    Rzt = mat([[cos(th[n-1, c]), -sin(th[n-1, c]), 0, 0],
               [sin(th[n-1, c]), cos(th[n-1, c]), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], copy=False)

    Rxa = mat([[1, 0, 0, 0],
               [0, cos(alph[0, n-1]), -sin(alph[0, n-1]), 0],
               [0, sin(alph[0, n-1]), cos(alph[0, n-1]), 0],
               [0, 0, 0, 1]], copy=False)

    A_i = T_d * Rzt * T_a * Rxa

    return A_i

def HTrans(th, c):
    A_1 = AH(1, th, c)
    A_2 = AH(2, th, c)
    A_3 = AH(3, th, c)
    A_4 = AH(4, th, c)
    A_5 = AH(5, th, c)
    A_6 = AH(6, th, c)

    T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6

    return T_06

# ************************************************** INVERSE KINEMATICS

def invKine(desired_pos):  # T60
    th = mat(np.zeros((6, 8)))
    P_05 = (desired_pos * mat([0, 0, -d6, 1]).T - mat([0, 0, 0, 1]).T)

    # print("P_05:", P_05)
   
    # **** theta1 ****

    psi = atan2(P_05[1, 0], P_05[0, 0])
    phi = float(np.arccos(float(d4) / float(np.sqrt( P_05[1, 0]**2 + P_05[0, 0]**2,dtype = np.float32))))
    # The two solutions for theta1 correspond to the shoulder being either left or right
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi
    th = th.real

    # print("theta1:", th[0])

    # **** theta5 ****

    cl = [0, 4]  # wrist up or down
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH(1, th, c))
        T_16 = T_10 * desired_pos
        print(T_16)
        th[4, c:c+2] = + acos((T_16[2, 3] - d4) / d6)
        th[4, c+2:c+4] = - acos((T_16[2, 3] - d4) / d6)

    th = th.real

    # print("theta5:", th[4])

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1, 3), T16(2, 3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = linalg.inv(T_10 * desired_pos)
        th[5, c:c+2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))

    th = th.real

    # print("theta6:", th[5])

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = AH(6, th, c)
        T_54 = AH(5, th, c)
        T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T
        t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2) / (2 * a2 * a3))  # norm ?
        th[2, c] = t3.real
        th[2, c+1] = -t3.real

    # print("theta3:", th[2])

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH(1, th, c))
        T_65 = np.linalg.inv(AH(6, th, c))
        T_54 = np.linalg.inv(AH(5, th, c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3 * sin(th[2, c]) / linalg.norm(P_13))
        # theta 4
        T_32 = linalg.inv(AH(3, th, c))
        T_21 = linalg.inv(AH(2, th, c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])

    th = th.real

    return th.T




TIME_STEP = 64

camera = Camera('camera')
range_finder = RangeFinder('range-finder')

range_finder.enable(TIME_STEP)
camera.enable(TIME_STEP)

## grab RGB frame and convert to numpy ndarray
def get_rgb_frame() -> np.ndarray :
    image_array = camera.getImageArray()
    np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    return np_image

## grab Depth frame and convert to numpy ndarray
# def get_depth_frame() -> np.ndarray :
    # image_array = range_finder.getRangeImageArray()
    # np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    # return np_image


class UR5e:
    def __init__(self , name="my_robot"):
        # get the motor devices
        m1 = robot.getDevice('shoulder_lift_joint')
        m2 = robot.getDevice('shoulder_pan_joint')
        m3 = robot.getDevice('elbow_joint')

        m4 = robot.getDevice('wrist_1_joint')
        m5 = robot.getDevice('wrist_2_joint')
        m6 = robot.getDevice('wrist_3_joint')

        self.vac = robot.getDevice('vacuum gripper')
        self.vac.enablePresence(1)

        self.motors_list = [m1,m2,m3,m4,m5,m6]

        self.gps = robot.getDevice('gps')
        self.gps.enable(1)

        sampling_period = 1

        for m in self.motors_list:
            m.getPositionSensor().enable(sampling_period)
            m.enableForceFeedback(sampling_period)
            m.enableTorqueFeedback(sampling_period)


    def set_arm_torques(self , torques):
        for i,motor in enumerate(self.motors_list):
            motor.setTorque(torques[i])

    def set_gripper_pos(self, state = 'on'):
        ''' state : set vacuum gripper "on" or "off" for vacuum activation'''
        if state == 'on' or state == 'On' or state == 'ON':
            self.vac.turnOn()
        else:
            self.vac.turnOff()

    def set_arm_pos(self , pos):
        for i,motor in enumerate(self.motors_list):
            motor.setPosition(pos[i])

    def get_arm_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.motors_list]
        return p

    def get_gripper_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.gripper_list]
        return p

    def get_EE_position(self):
        return self.gps.value
        
def delay(ms):
        initTime = robot.getTime()      # Store starting time (in seconds)
        while robot.step(TIME_STEP ) != -1:
            if (robot.getTime() - initTime) * 1000.0 > ms: # If time elapsed (converted into ms) is greater than value passed in
                break


######################################################################
            
model = YOLO("/home/dorsa/Robotics/ai-in-robotics-course-project-AUT2024-main/UR5e_Project/best(1).pt")


## robot instance
ur5 = UR5e()

t_ms = 0
img = get_rgb_frame()   

# depth = get_depth_frame()
# ur5.set_gripper_pos(state = 'off')


#CNN

a1 = [-1.94,0.23,-0.9 ,1.27,-1.6,-1.33]
ur5.set_arm_pos(a1)
img = get_rgb_frame() 

transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)

# Perform object detection
with torch.no_grad():
    results = model(img_tensor)

for result in results:
    # Get bounding boxes in [xmin, ymin, xmax, ymax] format
    boxes = result.boxes.xyxy.cpu().numpy()

    # Get class labels
    labels = [model.names[int(cls)] for cls in result.boxes.cls]

    # Get confidence scores
    confidences = result.boxes.conf.cpu().numpy()


arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype = np.float32)
arr = arr.reshape(7, 4)
# write the image bounding boxes
for i,(box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
    if label == "box":
      label = 1
      z_real = 0.1
    elif label == "soda":
      label = 2
      z_real = 0.15
    elif label == "mobile":
      label = 3
      z_real = 0.025
    elif label == "biscuite_box":
      label = 4
      z_real = 0.05
    elif label == "mouse":
      label = 5
      z_real = 0.04
    x, y, w, h = box[:4]
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)


    # Calculate center of bounding box
    center_x = (x + w)/ 2
    center_y = (y + h) / 2
    

    
    a = (1/2)-(center_x/640)
    b = (-1/2 )+(center_y/480)

    x_real = -(b*0.577)-0.45
    y_real = a*0.788
    
    
    arr[i][0] = label
    arr[i][1] = float(x_real)
    arr[i][2] = float(y_real)
    arr[i][3] = float(z_real)
    
    
    print(f"Box: {box}, Label: {label}, Confidence: {confidence},center_x: {center_x}, center_y: {center_y}")
print(arr)

if arr[0][0] != 0:
    for i,ar in enumerate(arr):
    
        delay(3000)
        x = ar[1]
        y = ar[2]
        z = ar[3]
    
        position = np.array([[1, 0, 0, x], [0, -1, 0, y], [0, 0, -1, z], [0, 0, 0, 1]],dtype = np.float32)
        ik_results = invKine(position)
        pos = np.take(ik_results, indices=[2], axis=0)
        if pos[0,0] >= 0 :
            pos[0,0] -= 3.14
        else:
            pos[0,0] += 3.14
    
        pos_3 = [pos[0,1], pos[0,0], pos[0,2], pos[0,3], pos[0,4], pos[0,5]]
        pos_2 = [-1.94, pos[0,0], pos[0,2], pos[0,3], pos[0,4], pos[0,5]]
        
        a1 = [-1.94,0.23,-0.9 ,1.27,-1.6,-1.33]
        ur5.set_arm_pos(a1)
        delay(2000)
        ur5.set_arm_pos(pos_2)
        delay(2000)
        ur5.set_arm_pos(pos_3)
        delay(2000)
        ur5.set_gripper_pos(state = 'on')
        delay(2000)
    
    
    
        ##############
        # 4,2 => Green
        # 1 => Blue
        # 5,3 => Red
        ##############
    
        
        if ar[0] == 1:
            #Pick : done
    
            #Place
            delay(1000) 
            Bbox = [ -1.1,1.3,1.2,-1.57,-1.57,0]  
            ur5.set_arm_pos(Bbox)
            delay(1000)  
            Bbox = [-0.5,1.3,1.6,-2.7,-1.57,0]
            ur5.set_arm_pos(Bbox)
            ur5.set_gripper_pos(state = 'off')
    
        elif ar[0] == 2:
            #Pick : done
    
            #Place
            delay(2000)
            Gbox = [ -1.1,0.1,1.2,-1.57,-1.57,0]  
            ur5.set_arm_pos(Gbox)
            delay(1000)      
            Gbox = [ -0.9,0.1,1.2,-1.57,-1.57,0]
            ur5.set_arm_pos(Gbox)
            ur5.set_gripper_pos(state = 'off')
    
        elif ar[0] == 3:
            #Pick : done
    
            #Place
            delay(2000)
            Rbox = [ -1.1,-0.5,1.2,-1.57,-1.57,0]  
            ur5.set_arm_pos(Rbox)
            delay(1000)
            Rbox = [ -0.9,-0.5,1.2,-1.57,-1.57,0]
            ur5.set_arm_pos(Rbox)
            ur5.set_gripper_pos(state = 'off')
            
        elif ar[0] == 4:
            #Pick : done
    
            #Place
            delay(2000)    
            Gbox = [ -1.1,0.1,1.2,-1.57,-1.57,0]  
            ur5.set_arm_pos(Gbox)
            delay(1000)
            Gbox = [ -0.9,0.1,1.2,-1.57,-1.57,0]
            ur5.set_arm_pos(Gbox)
            ur5.set_gripper_pos(state = 'off')
    
        elif ar[0] == 5:
            #Pick : done
    
            #Place
            delay(2000)
            Rbox = [ -1.1,-0.5,1.2,-1.57,-1.57,0]  
            ur5.set_arm_pos(Rbox)
            delay(1000)
            Rbox = [ -0.9,-0.5,1.2,-1.57,-1.57,0]
            ur5.set_arm_pos(Rbox)
            ur5.set_gripper_pos(state = 'off')

delay(2000)           
ur5.set_gripper_pos(state = 'off')
delay(2000)
