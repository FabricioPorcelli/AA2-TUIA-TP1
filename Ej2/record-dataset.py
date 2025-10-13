import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================

# Carpeta donde se guardarán los archivos .npy del dataset
DATA_PATH = "dataset"
os.makedirs(DATA_PATH, exist_ok=True)  # Crea la carpeta si no existe

# Diccionario con los gestos y sus etiquetas numéricas
GESTURES = {0: "Piedra", 1: "Papel", 2: "Tijeras"}

# ============================================================
# INICIALIZAR MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands

# Inicializa el modelo de detección y seguimiento de manos
hands = mp_hands.Hands(
    static_image_mode=False,       # Usa detección en modo video (no imagen fija)
    max_num_hands=1,               # Detecta solo una mano a la vez
    min_detection_confidence=0.5,  # Confianza mínima para detección
    min_tracking_confidence=0.5    # Confianza mínima para seguimiento
)

# Herramienta para dibujar los puntos y conexiones de la mano en pantalla
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# CONFIGURACIÓN DE LA CÁMARA
# ============================================================

cap = cv2.VideoCapture(0)  # Abre la cámara por defecto (índice 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Ancho del cuadro de video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto del cuadro de video

# ============================================================
# ARRAYS PARA GUARDAR LOS DATOS
# ============================================================

data = []    # Lista para guardar las coordenadas (x, y) de cada frame
labels = []  # Lista para guardar las etiquetas correspondientes (0, 1, 2)

print("Presiona '0' (piedra), '1' (papel), '2' (tijeras), 'q' para salir.")

# ============================================================
# BUCLE PRINCIPAL: CAPTURA Y PROCESAMIENTO DE FRAMES
# ============================================================

while True:
    ret, frame = cap.read()  # Lee un frame de la cámara
    if not ret:
        break  # Si no se pudo capturar, termina el bucle

    # Convierte la imagen de BGR (OpenCV) a RGB (MediaPipe trabaja en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa la imagen con MediaPipe para detectar la mano
    results = hands.process(frame_rgb)

    # Si se detectó una mano...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Dibuja los 21 puntos y sus conexiones sobre la imagen
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extrae las coordenadas normalizadas (x, y) de los 21 puntos de la mano
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])  # agrega 42 valores (x1,y1,x2,y2,...)

            # Muestra instrucciones en pantalla
            cv2.putText(frame, "0=Piedra | 1=Papel | 2=Tijeras | q=Salir",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            # Espera una tecla
            key = cv2.waitKey(1) & 0xFF

            # Si se presiona una tecla válida (0, 1 o 2)
            if key in [ord('0'), ord('1'), ord('2')]:
                label = int(chr(key))  # Convierte la tecla en número
                data.append(coords)    # Guarda los landmarks
                labels.append(label)   # Guarda la etiqueta
                print(f"Gesto guardado: {GESTURES[label]} (total={len(data)})")
                cv2.waitKey(300)  # Espera un poco para evitar duplicados

            # Si se presiona 'q', se guarda y sale
            elif key == ord('q'):
                # Guarda los arrays como archivos .npy en la carpeta dataset
                np.save(os.path.join(DATA_PATH, "rps_dataset.npy"), np.array(data))
                np.save(os.path.join(DATA_PATH, "rps_labels.npy"), np.array(labels))
                print("✅ Dataset guardado en carpeta 'dataset'")

                # Libera la cámara y cierra las ventanas
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Muestra el frame con los landmarks y las instrucciones
    cv2.imshow("Grabando gestos", frame)