import cv2
import mediapipe as mp
import URBasic
import time
import math3d as m3d
import math


# Constants for hand tracking
MIN_HAND_SIZE = 47
MAX_HAND_SIZE = 100 

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

# Initialize robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
time.sleep(1)

robot.movej(q=joint_position, a=ACCELERATION, v=VELOCITY)

robot_position = [0, 0, 0]
origin = None

robot.init_realtime_control()
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
    if hand_landmarks:
        # Posição da mão no centro (baseado nos pontos 1 e 17 da mão)
        hand_center = (
            (hand_landmarks[1].x + hand_landmarks[17].x) / 2,  # Centro X da mão
            (hand_landmarks[1].y + hand_landmarks[17].y) / 2,  # Centro Y da mão
        )

        # Distância da mão (escala do tamanho da mão)
        hand_distance_x = abs(hand_landmarks[1].x - hand_landmarks[17].x)
        hand_distance_y = abs(hand_landmarks[1].y - hand_landmarks[17].y)

        # Cálculo do tamanho da mão com base na distância entre pontos
        hand_size = (video_resolution[0] * (hand_distance_x**2 + hand_distance_y**2) ** 0.5)

        # Escalando a distância Z com base no tamanho da mão
        if(hand_size > MAX_HAND_SIZE):
            hand_distance_center_z = 600  # Distância maior
        elif(hand_size < MIN_HAND_SIZE):
            hand_distance_center_z = 1  # Distância menor
        else:
            # Distância Z escalada de forma linear
            hand_distance_center_z = (hand_size - MIN_HAND_SIZE) / (MAX_HAND_SIZE - MIN_HAND_SIZE) * video_resolution[0]

        # Calculando a posição em X e Y relativa ao centro da imagem
        hand_distance_center_x = hand_center[0] - (video_resolution[0] / 2)
        hand_distance_center_y = hand_center[1] - (video_resolution[1] / 2)

        # Convertendo as coordenadas 2D para o mundo 3D
        scale_x = 0.01  # Fator de escala para o movimento em X
        scale_y = 0.01  # Fator de escala para o movimento em Y
        scale_z = 0.01  # Fator de escala para o movimento em Z

        x_pos = hand_distance_center_x * scale_x
        y_pos = hand_distance_center_y * scale_y
        z_pos = hand_distance_center_z * scale_z

        move_joint_positions = initial_joint_positions[:]
        move_joint_positions[0] += scale_x
        move_joint_positions[1] += scale_y
        move_joint_positions[2] += scale_z

        # Calculando a posição final do robô
        robot_target_position = [robot_pos[0] + x_pos, robot_pos[1] + y_pos, robot_pos[2] + z_pos]

        # Corrigir valores para o limite máximo
        robot_target_position = check_max_xy(robot_target_position)

        # Movendo o braço robótico para a nova posição calculada
        robot.movej(q=move_joint_positions, a=ACCELERATION, v=VELOCITY)




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