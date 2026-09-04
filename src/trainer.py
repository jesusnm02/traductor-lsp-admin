import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import subprocess
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

os.makedirs(MODELOS_DIR, exist_ok=True)

class LSPTrainer:
    def __init__(self, dataset_manager: LSPDataManager, export_base_dir=None):
        """
        Clase encargada de entrenar las redes neuronales por categoría y exportarlas a TensorFlow.js.
        Usa rutas absolutas para persistencia bajo el directorio raíz.
        """
        self.dataset_manager = dataset_manager
        if export_base_dir is None:
            self.export_base_dir = MODELOS_DIR
        elif not os.path.isabs(export_base_dir):
            self.export_base_dir = os.path.join(ROOT_DIR, export_base_dir)
        else:
            self.export_base_dir = export_base_dir

        os.makedirs(self.export_base_dir, exist_ok=True)

    def build_model(self, num_classes, input_shape=(30, 255)):
        """
        Construye una arquitectura de red secuencial LSTM optimizada para ejecutarse
        en dispositivos móviles y navegadores web a través de TensorFlow.js de forma ligera.
        """
        model = Sequential([
            Input(shape=input_shape),
            
            # Primera capa LSTM con Batch Normalization para estabilizar el entrenamiento
            LSTM(64, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Segunda capa LSTM que reduce la secuencia a un vector plano
            LSTM(64, return_sequences=False, activation='tanh'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Capa densa intermedia para aprender combinaciones de características de landmarks
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Capa de salida con activación Softmax para clasificación de múltiples palabras
            Dense(num_classes, activation='softmax')
        ])
        
        # Usar el optimizador Adam con una tasa de aprendizaje moderada
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_category(self, category_name: str, epochs=100, batch_size=16, validation_split=0.2):
        """
        Carga el historial completo de coordenadas, entrena la red neuronal LSTM y exporta
        el modelo en el formato Keras nativo (.keras).
        
        Args:
            category_name (str): Nombre de la categoría (ej. "numeros").
            epochs (int): Número máximo de iteraciones de entrenamiento.
            batch_size (int): Cantidad de muestras por lote.
            validation_split (float): Porcentaje de datos reservados para validación interna.
            
        Returns:
            history: El historial del entrenamiento con las métricas de precisión y pérdida.
            export_dir: Ruta donde se exportaron los resultados.
        """
        # 1. Cargar el dataset completo (Asegura un entrenamiento libre de olvido catastrófico)
        X, y, label_map = self.dataset_manager.load_dataset_for_training(category_name)
        num_classes = len(label_map)
        input_shape = (X.shape[1], X.shape[2])
        
        print(f"\n[ENTRENAMIENTO] Iniciando reentrenamiento completo de la categoría: {category_name.upper()}")
        print(f"[ENTRENAMIENTO] Clases detectadas ({num_classes}): {list(label_map.values())}")
        print(f"[ENTRENAMIENTO] Total de muestras cargadas en disco: {X.shape[0]}")
        print(f"[ENTRENAMIENTO] Dimensión de entrada adaptada dinámicamente: {input_shape}")

        # 2. Dividir en conjuntos de entrenamiento y prueba para evaluar sobreajuste
        # Se aplica un random_state constante para la reproducibilidad de los resultados de la tesis
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            stratify=y, 
            random_state=42
        )

        # 3. Construir el modelo adaptado dinámicamente al número de clases y dimensiones de entrada
        model = self.build_model(num_classes=num_classes, input_shape=input_shape)
        model.summary()

        # 4. Callbacks avanzados para controlar el sobreajuste (Overfitting)
        callbacks = [
            # Detiene el entrenamiento si la pérdida de validación deja de mejorar durante 15 épocas
            EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce la tasa de aprendizaje si el entrenamiento se estanca
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=7, 
                min_lr=1e-5,
                verbose=1
            )
        ]

        # 5. Ejecutar el entrenamiento de la red
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # 6. Crear carpetas de destino para la exportación
        category_export_path = os.path.join(self.export_base_dir, category_name.lower().strip())
        if not os.path.exists(category_export_path):
            os.makedirs(category_export_path)

        # Guardar archivo de metadatos/etiquetas (labels.json) indispensable para la web del alumno
        # Formato: {"0": "palabra_1", "1": "palabra_2"}
        labels_json_path = os.path.join(category_export_path, "labels.json")
        labels_dict = {str(k): v for k, v in label_map.items()}
        with open(labels_json_path, 'w', encoding='utf-8') as f:
            json.dump(labels_dict, f, indent=4, ensure_ascii=False)

        # Guardar el modelo Keras temporalmente
        keras_model_path = os.path.join(category_export_path, "model.keras")
        model.save(keras_model_path)
        print(f"[ENTRENAMIENTO] Modelo guardado localmente en Keras: {keras_model_path}")

        # 7. Convertir automáticamente a formato TensorFlow.js para la PWA del alumno
        tfjs_export_path = os.path.join(category_export_path, "tfjs_model")
        self._convert_to_tfjs(keras_model_path, tfjs_export_path)

        return history, tfjs_export_path

    def _convert_to_tfjs(self, keras_model_path, tfjs_export_path):
        """
        Llama al compilador de tensorflowjs para fragmentar y serializar el modelo .keras
        en un archivo model.json y fragmentos binarios (.bin) compatibles con la web del alumno.
        """
        print(f"[CONVERTIDOR] Iniciando conversión de '{keras_model_path}' a formato TensorFlow.js...")
        
        if not os.path.exists(tfjs_export_path):
            os.makedirs(tfjs_export_path)

        # Comando CLI oficial de TensorFlow.js Converter para empaquetado optimizado
        command = [
            "tensorflowjs_converter",
            "--input_format=keras",
            keras_model_path,
            tfjs_export_path
        ]

        try:
            # Ejecutar el comando en el sistema operativo
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"[CONVERTIDOR] Conversión exitosa. Archivos generados en: {tfjs_export_path}")
        except subprocess.CalledProcessError as e:
            print(f"[CONVERTIDOR] ERROR al ejecutar la conversión por CLI: {e.stderr}")
            # Si falla la llamada por CLI, usamos la API programática nativa de respaldo
            self._convert_to_tfjs_programmatic(keras_model_path, tfjs_export_path)
        except Exception as e:
            print(f"[CONVERTIDOR] ERROR inesperado: {str(e)}")

    def _convert_to_tfjs_programmatic(self, keras_model_path, tfjs_export_path):
        """Alternativa programática en Python por si falla el comando de consola subprocess."""
        try:
            import tensorflowjs as tfjs
            model = tf.keras.models.load_model(keras_model_path)
            tfjs.converters.save_keras_model(model, tfjs_export_path)
            print(f"[CONVERTIDOR - RESPALDO] Conversión programática realizada con éxito en: {tfjs_export_path}")
        except Exception as e:
            print(f"[CONVERTIDOR - RESPALDO] ERROR Crítico: No se pudo realizar la conversión programática. {str(e)}")
            print("Por favor, asegúrate de tener instalado 'tensorflowjs' en tu entorno virtual: pip install tensorflowjs")