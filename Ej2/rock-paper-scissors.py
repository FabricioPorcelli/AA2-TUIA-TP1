import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# CARGA DEL MODELO ENTRENADO
# ============================================================

# Se carga el modelo previamente entrenado y guardado como "rps_model.keras"
# Este modelo fue entrenado con las coordenadas de los landmarks de la mano.
model = tf.keras.models.load_model("./Ej2/rps_model.keras")

# ============================================================
# CONFIGURACIÓN DE MEDIAPIPE HANDS
# ============================================================

# MediaPipe Hands es el modelo de detección y seguimiento de manos de Google.
# Permite obtener 21 puntos clave (landmarks) por cada mano detectada.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicialización del detector de manos
hands = mp_hands.Hands(
    static_image_mode=False,          # Detección continua en video
    max_num_hands=1,                  # Detectar solo una mano
    min_detection_confidence=0.7,     # Confianza mínima para detección
    min_tracking_confidence=0.7       # Confianza mínima para seguimiento
)

# ============================================================
# DEFINICIÓN DE CLASES
# ============================================================

# Las tres categorías del juego
CLASSES = ["Piedra", "Papel", "Tijeras"]

# ============================================================
# CAPTURA DE VIDEO EN TIEMPO REAL
# ============================================================

# Se activa la cámara web (índice 0 por defecto)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Si no se obtiene imagen, se interrumpe el bucle

    # Espejar la imagen para que se vea como un reflejo (más intuitivo)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe requiere imágenes en formato RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Texto por defecto si no se detecta mano
    prediction_text = "Esperando mano..."

    # ========================================================
    # DETECCIÓN DE MANO Y CLASIFICACIÓN
    # ========================================================

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los landmarks y conexiones sobre la imagen
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extraer las coordenadas normalizadas (x, y) de los 21 puntos
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Convertir a array NumPy y dar forma (1, 42)
            scaler = joblib.load("./Ej2/scaler.pkl")
            X = np.array(landmarks).reshape(1, -1)
            X = scaler.transform(X)                 # IMPORTANTE: Usar el mismo scaler del entrenamiento
 
            # Realizar predicción con el modelo entrenado
            prediction = model.predict(X)
            pred_class = np.argmax(prediction)  # Clase con mayor probabilidad
            prob = np.max(prediction)           # Nivel de confianza

            # Mostrar resultado si la confianza es alta
            if prob > 0.7:
                prediction_text = f"{CLASSES[pred_class]} ({prob*100:.1f}%)"
            else:
                prediction_text = "Inseguro"

    # ========================================================
    # VISUALIZACIÓN EN PANTALLA
    # ========================================================

    # Mostrar el texto de la predicción sobre el frame
    cv2.putText(
        frame, prediction_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 3
    )

    # Mostrar ventana con el resultado
    cv2.imshow("Piedra, Papel o Tijeras", frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# LIBERAR RECURSOS
# ============================================================

# Cerrar cámara y ventana al finalizar
cap.release()
cv2.destroyAllWindows()