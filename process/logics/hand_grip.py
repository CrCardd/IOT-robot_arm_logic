import cv2 
import mediapipe as mp
import time

# Constants for hand tracking
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 400
SCREEN_CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

# Initialize MediaPipe Hands module
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


vs = cv2.VideoCapture(0) 




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

    h, w, _ = frame.shape

    hand_center = (
        int((((hand_landmarks[1].x + hand_landmarks[17].x) / 2 * w) + ((hand_landmarks[0].x + hand_landmarks[5].x) / 2 * w)) / 2),  
        int((((hand_landmarks[1].y + hand_landmarks[17].y) / 2 * h) + ((hand_landmarks[0].y + hand_landmarks[5].y) / 2 * h)) / 2),
    )

    hand_distance_size_x = w * abs(hand_landmarks[1].x - hand_landmarks[17].x)
    hand_distance_size_y = h * abs(hand_landmarks[1].y - hand_landmarks[17].y)
    hand_size = (hand_distance_size_x**2 + hand_distance_size_y**2) ** 0.5

    cv2.putText(frame, str("(0)"), (hand_center[0], hand_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    hand_grip = 0
    for i in range(8,21,4):
        if i < 15:
            point = 1
        else:
            point = 0
        


        crr_center_x = int((w * (hand_landmarks[i-2].x + hand_landmarks[point].x)) / 2)
        crr_center_y = int((h * (hand_landmarks[i-2].y + hand_landmarks[point].y)) / 2)

        finger_distance_x = abs(crr_center_x - (w * hand_landmarks[i].x))
        finger_distance_y = abs(crr_center_y - (h * hand_landmarks[i].y))

        finger_distance = (finger_distance_x*finger_distance_x + finger_distance_y*finger_distance_y) ** 0.5
        hand_grip += finger_distance

        cv2.putText(frame, str("."), (crr_center_x, crr_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,50,255 - finger_distance), 2)
        cv2.line(frame, (int(w * hand_landmarks[i].x), int(h * hand_landmarks[i].y)), (crr_center_x, crr_center_y), (20,finger_distance,280 - finger_distance), 1)  #AAAAAAAAAAAAAAAAAAAAAA
    hand_grip /= 4

    hand_grip = (int((1 - hand_grip / (hand_size * 1) ) * 100))

    if(hand_grip < 3):
        hand_grip = 3
    elif(hand_grip > 97):
        hand_grip = 97

    hand_grip = int(hand_grip / 10) * 10

 
    hand_top_distance_x = abs(hand_landmarks[5].x - hand_landmarks[17].x)
    hand_top_distance_y = abs(hand_landmarks[5].y - hand_landmarks[17].y)
    hand_top_distance = w * ((hand_top_distance_x*hand_top_distance_x) + (hand_top_distance_y*hand_top_distance_y)) ** 0.5

    # print(hand_top_distance)
    # print(hand_size)
    # print()

    hand_top_normal_distance = hand_size * 0.7
    if not hand_top_distance > hand_top_normal_distance:

        rotate_wrist = hand_top_distance / hand_top_normal_distance

        if hand_landmarks[5].z > hand_landmarks[17].z:
            rotate_wrist *= 1
        elif hand_landmarks[17].z > hand_landmarks[5].z:
            rotate_wrist *= -1
    else:
        rotate_wrist = 0


    print(rotate_wrist)



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