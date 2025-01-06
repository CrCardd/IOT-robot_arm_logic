import cv2
import mediapipe as mp
# import URBasic
import time
import math3d as m3d
import math


# Constants for hand tracking
# MAX_DIST_Z = 0.24
# MIN_DIST_Z = 0.24

# Initialize MediaPipe Hands module
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Initialize robot settings
ROBOT_IP = '169.254.41.22'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# Initial joint positions (in radians)
joint_position = [-1.7075, -1.4654, -1.5655, -0.1151, 1.5962, -0.0105]

video_resolution = (700, 400)
video_midpoint = (int(video_resolution[0] / 2), int(video_resolution[1] / 2))

vs = cv2.VideoCapture(0)  # OpenCV video capture
vs.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])


print("initialising robot")
# robotModel = URBasic.robotModel.RobotModel()
# robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

# robot.reset_error()
print("robot initialised")
time.sleep(1)

# robot.movej(q=joint_position, a=ACCELERATION, v=VELOCITY)

robot_position = [0, 0, 0]
# origin = None

# robot.init_realtime_control()
time.sleep(1)



def find_hands(image):
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Hands.process(frame_rgb)
    hand_points = results.multi_hand_landmarks
    if hand_points:
        for points in hand_points:
            mpDraw.draw_landmarks(image, points, hands.HAND_CONNECTIONS)
    return hand_points, image


def move_to_hand(hand_landmarks, robot_pos):
    
    hand_center = (
        (hand_landmarks[1].x + hand_landmarks[1].x) / 2, 
        (hand_landmarks[1].y + hand_landmarks[1].y) / 2
    )
    hand_distance_x = abs(hand_landmarks[1].x - hand_landmarks[17].x)
    hand_distance_y = abs(hand_landmarks[1].y - hand_landmarks[17].y)
    hand_size = (hand_distance_x*hand_distance_x + hand_distance_y*hand_distance_y) ** 0.5


    scale_x = 2.5   
    scale_y = 2.5   
    scale_z = 2.5


    x_pos = hand_center[0] * video_resolution[0]
    y_pos = hand_center[1] * scale_y
    z_pos = hand_size      * video_resolution[0]

    print(x_pos, z_pos)    
    # robot_target_position = [robot_pos[0] + x_pos, robot_pos[1] + y_pos, robot_pos[2] + z_pos]
    # robot_target_position = check_max_xy(robot_target_position)

    



    # joint_position[0] += x_pos   
    # joint_position[1] += y_pos
    # joint_position[2] += z_pos

    # robot.movej(q=joint_position, a=ACCELERATION, v=VELOCITY)


def check_max_xy(robot_target_xy):
    max_dist = 3  # Increase maximum distance for X and Y
    robot_target_xy[0] = max(-max_dist, min(robot_target_xy[0], max_dist))
    robot_target_xy[1] = max(-max_dist, min(robot_target_xy[1], max_dist))
    return robot_target_xy



while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    hand_points, frame = find_hands(frame)

    results = Hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        move_to_hand(hand_landmarks, robot_position)




    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()