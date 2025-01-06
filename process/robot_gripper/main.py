import cv2 
import mediapipe as mp
import time
from gripper import Gripper

# Constants for hand tracking
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 400
SCREEN_CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)


IP = '169.254.108.43'
gripper = Gripper(IP)

# Initialize MediaPipe Hands module
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

video_resolution = (SCREEN_WIDTH, SCREEN_HEIGHT)
video_midpoint = (int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2))

vs = cv2.VideoCapture(0)  # OpenCV video capture
vs.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])




def find_hands(image):
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Hands.process(frame_rgb)
    hand_points = results.multi_hand_landmarks
    if hand_points:
        for points in hand_points:
            mpDraw.draw_landmarks(image, points, hands.HAND_CONNECTIONS)
    
    return hand_points, image



# Function to control robot based on hand position
def move_to_hand(hand_landmarks, frame):
    if not hand_landmarks:
        return

    hand_center = (
        int((hand_landmarks[1].x + hand_landmarks[17].x) / 2 * video_resolution[0]),  
        int((hand_landmarks[1].y + hand_landmarks[17].y) / 2 * video_resolution[1]),
    )

    screen_y = hand_center[1]
    screen_x = hand_center[0]

    hand_distance_x = abs(hand_landmarks[1].x - hand_landmarks[17].x)
    hand_distance_y = abs(hand_landmarks[1].y - hand_landmarks[17].y)
    hand_size = (video_resolution[0] * (hand_distance_x**2 + hand_distance_y**2) ** 0.5)

    hand_grip = 0
    for i in range(8,21,4):
        if i < 9:
            point = 0
        else:
            point = 1

        crr_center_x = int((video_resolution[0] * (hand_landmarks[i-3].x + hand_landmarks[point].x)) / 2)
        crr_center_y = int((video_resolution[1] * (hand_landmarks[i-3].y + hand_landmarks[point].y)) / 2)

        finger_distance_x = abs(crr_center_x - (video_resolution[0] * hand_landmarks[i].x))
        finger_distance_y = abs(crr_center_y - (video_resolution[1] * hand_landmarks[i].y))

        finger_distance = (finger_distance_x*finger_distance_x + finger_distance_y*finger_distance_y) ** 0.5
        hand_grip += finger_distance

        cv2.putText(frame, str("O"), (crr_center_x, crr_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.line(frame, (int(video_resolution[0] * hand_landmarks[i].x), int(video_resolution[1] * hand_landmarks[i].y)), (crr_center_x, crr_center_y), (100,255,100), 1)  #AAAAAAAAAAAAAAAAAAAAAA
    hand_grip /= 4

    # hand_grip = int((1 - (hand_grip / (hand_size * 1.5) ) ) * 94 + 3)
    hand_grip = (int((1 - hand_grip / (hand_size * 1.23) ) * 100))
    hand_grip = int(int(hand_grip / 19.5 + 1) * 19.5) 

    if(hand_grip < 3):
        hand_grip = 3
    elif(hand_grip > 97):
        hand_grip = 97


    print(hand_grip)
    gripper.set_position(hand_grip)



# Main loop for video capture and hand tracking
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    hand_points, frame = find_hands(frame)

    results = Hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        move_to_hand(hand_landmarks, frame)


        

    # Show the processed frame
    cv2.imshow("Hand Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
vs.release()
cv2.destroyAllWindows()