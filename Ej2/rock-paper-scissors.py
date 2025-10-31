import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# ============================================================
# CARGA DEL MODELO ENTRENADO
# ============================================================

model = tf.keras.models.load_model("./rps_model.keras")
scaler = joblib.load("./scaler.pkl")

# ============================================================
# CONFIGURACIÓN DE MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================
# DEFINICIÓN DE CLASES Y COLORES
# ============================================================

CLASSES = ["Piedra", "Papel", "Tijeras"]

# Colores asociados a cada mano detectada (BGR)
HAND_COLORS = [
    (100, 30, 30),
    (30, 30, 110)
]

# ============================================================
# CAPTURA DE VIDEO EN TIEMPO REAL
# ============================================================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predictions = []  # almacenará predicciones por mano

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            color = HAND_COLORS[i % len(HAND_COLORS)]

            # Dibujar landmarks y conexiones con el color asignado
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )

            # Extraer coordenadas (x, y) de los 21 puntos
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # Normalización espacial idéntica al dataset
            base = landmarks[0]  # muñeca (landmark 0)
            landmarks -= base

            max_range = np.max(np.linalg.norm(landmarks, axis=1))
            if max_range > 0:
                landmarks /= max_range

            # Aplanar para el modelo
            coords = landmarks.flatten().reshape(1, -1)

            # Escalar con el StandardScaler entrenado
            X = scaler.transform(coords)

            # Predicción
            prediction = model.predict(X, verbose=0)
            pred_class = np.argmax(prediction)
            prob = np.max(prediction)
            predictions.append((i, pred_class, prob, color))

    # ========================================================
    # VISUALIZACIÓN DE RESULTADOS
    # ========================================================

    font_scale = 0.8
    thickness = 2

    if predictions:
        for i, pred_class, prob, color in predictions:
            text = f"{CLASSES[pred_class]} ({prob*100:.1f}%)" if prob > 0.7 else "Inseguro"

            # Determinar posición del texto
            if i == 0:
                position = (10, 40)              # esquina superior izquierda
            else:
                position = (w - 250, 40)         # esquina superior derecha

            cv2.putText(frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    else:
        cv2.putText(frame, "Esperando manos...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Piedra, Papel o Tijeras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()