## Contexto del Proyecto
El sistema presenta un error crítico de congelamiento en el renderizado de la cámara mediante Flet. Además, se requiere implementar el pipeline de entrenamiento utilizando una CNN 1D para las 30 secuencias de landmarks normalizados, y un módulo de pruebas en tiempo real para validar los modelos generados por categoría.

## Tareas a Ejecutar

### 1. Fix Crítico: Congelamiento de Cámara (Flet UI)
**Archivo Objetivo:** Módulo encargado del renderizado de video (`vision_service.py` o donde se asigne `src_base64`).
**Problema:** La interfaz no se refresca asíncronamente durante el ciclo de lectura de OpenCV.
**Acción:** Inyectar la llamada a `update()` en el componente `Image` de Flet inmediatamente después de actualizar su propiedad `src_base64`.

```python
# FIX OBLIGATORIO en el loop de actualización del frame:
self.video_image_control.src_base64 = b64_string
self.video_image_control.update() # Refresco explícito del hilo secundario al Main Thread
2. Módulo de Entrenamiento: CNN Espacio-Temporal
Archivo Objetivo: Crear src/model_trainer.py e integrarlo a la UI (ui_components.py / app.py).
Interfaz: Añadir un botón "Generar Modelo de Categoría" que se habilite al tener al menos 30 muestras de cada palabra en la categoría seleccionada.
Acción: Implementar la arquitectura Convolucional 1D descrita a continuación para capturar las relaciones espaciales-visuales.

Python
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

class ModelTrainer:
    def __init__(self, sequence_length=30, features=111): # Ajustar features según normalizador (ej: 37*3)
        self.sequence_length = sequence_length
        self.features = features

    def build_and_train_cnn(self, X_train, y_train, num_classes, category_name):
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.sequence_length, self.features)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Entrenamiento
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=16)
        
        # Guardar modelo
        os.makedirs('modelos', exist_ok=True)
        model_path = f'modelos/modelo_LSP_{category_name}.h5'
        model.save(model_path)
        return model_path
3. Módulo de Pruebas y Validación en Vivo
Archivo Objetivo: Crear src/tester_service.py e integrarlo como una nueva pestaña/vista en la UI.
Interfaz:

Selector (Dropdown) de Categoría.

Botón "Iniciar Prueba" / "Detener Prueba".

Componente Text (ej. lbl_prediction) con tipografía grande para mostrar el resultado en vivo.
Acción: Implementar inferencia mediante ventana deslizante (sliding window) sobre el hilo de video.

Python
import numpy as np
from tensorflow.keras.models import load_model

class LiveTester:
    def __init__(self, model_path, labels):
        self.model = load_model(model_path)
        self.labels = labels
        self.sequence = []

    def process_frame(self, normalized_landmarks, prediction_label_control):
        self.sequence.append(normalized_landmarks)
        self.sequence = self.sequence[-30:] # Mantener buffer de 30 frames
        
        if len(self.sequence) == 30:
            # Inferencia
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            predicted_index = np.argmax(res)
            confidence = res[predicted_index]
            
            # Control de umbral y UI
            if confidence > 0.85:
                prediction_label_control.value = f"Seña: {self.labels[predicted_index]} ({confidence:.1%})"
                prediction_label_control.update() # Actualización asíncrona requerida