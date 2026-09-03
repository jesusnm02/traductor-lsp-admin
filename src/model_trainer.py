import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    """
    Módulo de Entrenamiento: Red Neuronal Convolucional 1D (CNN Espacio-Temporal).
    Captura relaciones espaciales y temporales de las 30 secuencias de landmarks normalizados.
    Exporta de forma unificada bajo 'data/modelos/{category}/' con formato nativo 'model.keras'.
    """
    def __init__(self, dataset_manager=None, sequence_length=30, features=255, export_base_dir="data/modelos"):
        self.dataset_manager = dataset_manager
        self.sequence_length = sequence_length
        self.features = features
        self.export_base_dir = export_base_dir
        os.makedirs(self.export_base_dir, exist_ok=True)

    def build_and_train_cnn(self, X_train, y_train, num_classes, category_name, epochs=50, batch_size=16, label_map=None):
        """
        Construye, compila y entrena la CNN 1D para la categoría especificada.
        Exporta el modelo entrenado a 'data/modelos/{category_name}/model.keras'
        y el mapa de etiquetas a 'data/modelos/{category_name}/labels.json'.
        """
        if X_train.ndim == 3:
            self.sequence_length = X_train.shape[1]
            self.features = X_train.shape[2]

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
        
        # Conversión a One-Hot Encoding si las etiquetas son enteros
        if y_train.ndim == 1 or y_train.shape[-1] != num_classes:
            y_cat = to_categorical(y_train, num_classes=num_classes)
        else:
            y_cat = y_train

        # Entrenamiento
        history = model.fit(X_train, y_cat, epochs=epochs, validation_split=0.2, batch_size=batch_size)
        
        # Guardar de forma unificada bajo data/modelos/{category}/model.keras
        category_export_path = os.path.join(self.export_base_dir, category_name.lower().strip())
        os.makedirs(category_export_path, exist_ok=True)
        keras_model_path = os.path.join(category_export_path, "model.keras")
        
        # Guardar de forma nativa en Keras
        model.save(keras_model_path)
        print(f"[ENTRENAMIENTO] Modelo guardado en formato moderno: {keras_model_path}")

        # Guardar mapa de etiquetas asociado para inferencia en tiempo real
        if label_map:
            labels_path = os.path.join(category_export_path, "labels.json")
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(label_map, f, indent=4, ensure_ascii=False)
            print(f"[ENTRENAMIENTO] Etiquetas guardadas en: {labels_path}")

        return keras_model_path

# Alias de compatibilidad para unificación en app.py
LSPTrainer = ModelTrainer
