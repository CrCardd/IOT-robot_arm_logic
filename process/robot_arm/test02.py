import URBasic
import rtde_control
import rtde_receive
import time

# Definindo o IP do robô
ROBOT_IP = '169.254.41.22'

# Conectando ao robô com URBasic (controle de movimento)
robotModel = URBasic.robotModel.RobotModel()
# robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

# Inicializando controle de movimento (URBasic)
# robot.reset_error()
# robot.init_realtime_control()
time.sleep(1)

# Conectando ao robô com RTDE para monitoramento em tempo real
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# Função para mover o robô usando URBasic
def move_robot():
    # Movimento do robô usando URBasic
    joint_positions = [0.5, 0.0, 0.1, 0.0, 1.57, 0.0]  # Exemplo de posição
    # robot.movej(joint_positions, a=0.5, v=0.5)
    print("Comando de movimento enviado com URBasic!")

# Função para monitorar as posições das juntas com RTDE
def monitor_robot():
    while True:
        # Obter as posições das juntas
        joint_positions = rtde_r.getActualQ()  # Obtém a posição das juntas em tempo real
        print(f"Posições das Juntas (em tempo real): {joint_positions}")
        
        # Verificar se o robô está em erro ou em um estado de segurança
        safety_mode = rtde_r.getSafetyMode()  # Obtém o modo de segurança do robô
        print(f"Modo de Segurança: {safety_mode}")

        # Adicionar um pequeno atraso para reduzir a carga no loop
        time.sleep(1)

# Executando o controle de movimento e monitoramento em tempo real
move_robot()  # Enviar comando de movimento ao robô com URBasic
monitor_robot()  # Monitorar o robô em tempo real com RTDE
