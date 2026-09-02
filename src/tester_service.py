import os
import json
import numpy as np
from tensorflow.keras.models import load_model

class LiveTester:
    """
    Módulo de Pruebas y Validación en Vivo con Despacho Seguro (Thread-Safe).
    Ejecuta inferencia mediante ventana deslizante (sliding window de 30 frames)
    sobre el flujo de video en tiempo real sin bloquear el hilo principal de Flet.
    """
    def __init__(self, model_path: str, labels, page_ref=None):
        """
        Args:
            model_path (str): Ruta al archivo .h5 o .keras del modelo.
            labels (list o dict): Mapeo de índices a nombres de señas.
            page_ref: Referencia opcional a la página de Flet para despacho seguro.
        """
        self.model_path = model_path
        self.model = load_model(model_path)
        self.page = page_ref
        
        # Normalizar labels a lista indexable
        if isinstance(labels, dict):
            self.labels = [labels[k] if k in labels else labels[str(k)] for k in range(len(labels))]
        else:
            self.labels = list(labels)
            
        self.sequence = []
        self.is_active = False
        self.last_prediction = ""
        self.last_confidence = 0.0

    def start(self):
        """Activa el modo de inferencia en tiempo real y reinicia el buffer."""
        self.sequence = []
        self.is_active = True

    def stop(self):
        """Detiene la inferencia en tiempo real."""
        self.is_active = False
        self.sequence = []

    def process_frame(self, normalized_landmarks, prediction_label_control=None, page_ref=None):
        """
        Procesa cada frame normalizado, mantiene la ventana de 30 frames y
        ejecuta la predicción cuando el buffer está lleno.
        Despacha la actualización de la UI a través de page.run_thread si está disponible.
        """
        if not self.is_active:
            return None, 0.0

        self.sequence.append(normalized_landmarks)
        self.sequence = self.sequence[-30:] # Mantener buffer de 30 frames
        
        if len(self.sequence) == 30:
            input_tensor = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_tensor, verbose=0)[0]
            predicted_index = int(np.argmax(res))
            confidence = float(res[predicted_index])
            
            self.last_confidence = confidence
            word = self.labels[predicted_index] if predicted_index < len(self.labels) else f"Clase_{predicted_index}"
            page = page_ref if page_ref is not None else self.page

            # Control de umbral (> 85%) y actualización segura de UI
            if confidence > 0.85:
                self.last_prediction = word
                if prediction_label_control:
                    def _update_ui():
                        prediction_label_control.value = f"Seña: {word.upper()} ({confidence:.1%})"
                        try:
                            prediction_label_control.update()
                        except Exception:
                            pass

                    if page and hasattr(page, "run_thread"):
                        page.run_thread(_update_ui)
                    else:
                        _update_ui()

                return word, confidence
            else:
                if prediction_label_control and not self.last_prediction:
                    def _update_ui_detecting():
                        prediction_label_control.value = f"Analizando movimiento... ({confidence:.1%})"
                        try:
                            prediction_label_control.update()
                        except Exception:
                            pass

                    if page and hasattr(page, "run_thread"):
                        page.run_thread(_update_ui_detecting)
                    else:
                        _update_ui_detecting()

                return None, confidence

        return None, 0.0
