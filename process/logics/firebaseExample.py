import firebase_admin
from firebase_admin import credentials, db

# Caminho para o arquivo JSON da chave privada do Firebase
cred = credentials.Certificate("C:/Users/crist/OneDrive/Área de Trabalho/Repositories/HandsRecognizeCode/src/robotarmtest-firebase-adminsdk-celpj-e13d4bf237.json")

# Inicializa o SDK do Firebase
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://robotarmtest-default-rtdb.firebaseio.com'
})

# Referência ao Realtime Database
ref = db.reference('robot_data')

# Escrever dados no Realtime Database
ref.set({
    'status': {
        'kau_trampa': '',
        'kau_trampa': '',
        'kau_trampa': '',
        'kau_trampa': '',
        'kau_trampa': '',
        'kau_trampa': '',
    }
})

# Ler dados do Realtime Database
usuarios = ref.get()
print(usuarios)
