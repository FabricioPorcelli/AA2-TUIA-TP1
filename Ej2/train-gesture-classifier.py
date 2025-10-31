import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ============================================================
# CARGA DEL DATASET
# ============================================================

X = np.load("dataset/rps_dataset.npy")
y = np.load("dataset/rps_labels.npy")

print("Dataset cargado:", X.shape, y.shape)

# ============================================================
# PREPROCESAMIENTO DEL DATASET
# ============================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl") # Se guarda el scaler para utilizarlo en el script de predicción

y_cat = to_categorical(y, num_classes=3)

# Dividimos manualmente para evitar fugas de datos
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# ============================================================
# DEFINICIÓN DEL MODELO
# ============================================================

model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(16, activation='relu'),
    Dropout(0.3),

    Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# GUARDADO DEL MODELO
# ============================================================

model.save("rps_model.keras")
print("Modelo entrenado y guardado como rps_model.keras")