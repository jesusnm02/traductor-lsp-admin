import os
import json
import time
import threading
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

def speak_word_offline(word: str):
    """
    Pronuncia la palabra traducida utilizando pyttsx3 de forma 100% offline y local en Windows.
    Se ejecuta en un hilo aislado para no bloquear el flujo de video ni la UI.
    """
    def _tts_thread():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Velocidad de habla natural y comprensible
            engine.say(word)
            engine.runAndWait()
        except Exception as e:
            print(f"Error en síntesis de voz TTS (pyttsx3): {str(e)}")

    threading.Thread(target=_tts_thread, daemon=True).start()

class LiveTester:
    """
    Módulo de Pruebas y Validación en Vivo con Debounce e Inferencia con Voz (TTS).
    - Ventana deslizante (30 frames) con normalización continua.
    - Confirmación sostenida por al menos 10 frames consecutivos con confianza > 85%.
    - Actualización de tipografía gigante e indicador de barra de confianza.
    - Síntesis de voz offline mediante pyttsx3 con regla de bloqueo de repetición.
    """
    def __init__(self, model_path: str, labels, page_ref=None):
        """
        Args:
            model_path (str): Ruta al archivo binario .keras o .h5 del modelo.
            labels (list o dict o str): Mapeo de índices a palabras o ruta a labels.json.
            page_ref: Referencia a la página Flet para despacho seguro de hilos.
        """
        # 1. Validación defensiva estricta de formato de modelo binario
        if not (model_path.endswith('.keras') or model_path.endswith('.h5')):
            # Detectar si se pasaron los argumentos invertidos por error
            if model_path.endswith('.json') and isinstance(labels, str) and (labels.endswith('.keras') or labels.endswith('.h5')):
                model_path, labels = labels, model_path
            else:
                raise ValueError(f"Ruta de modelo inválida: {model_path}. Debe ser un archivo binario .keras o .h5")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo binario: {model_path}")

        # 2. Cargar estrictamente el modelo binario con la API de Keras
        self.model_path = model_path
        self.model = load_model(model_path)
        self.page = page_ref
        
        # 3. Carga independiente de etiquetas (Labels)
        if isinstance(labels, str) and labels.endswith('.json') and os.path.exists(labels):
            with open(labels, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                if isinstance(label_data, dict):
                    self.labels = [label_data[str(i)] if str(i) in label_data else label_data[i] for i in range(len(label_data))]
                elif isinstance(label_data, list):
                    self.labels = label_data
                else:
                    self.labels = list(label_data)
        elif isinstance(labels, dict):
            self.labels = [labels[k] if k in labels else labels[str(k)] for k in range(len(labels))]
        elif isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        else:
            self.labels = []
            
        self.sequence = []
        self.is_active = False

        # Variables de Debounce e Inferencia
        self.candidate_word = ""
        self.consecutive_frames = 0
        self.min_consecutive_frames = 10  # Mínimo 10 frames consecutivos
        self.confidence_threshold = 0.85  # Umbral > 85%
        
        # Regla de bloqueo de repetición de voz
        self.last_spoken_word = ""
        self.last_spoken_time = 0.0
        self.no_gesture_count = 0

        # Últimas métricas
        self.last_prediction = ""
        self.last_confidence = 0.0

    def start(self):
        """Activa el modo de inferencia y resetea buffers."""
        self.sequence = []
        self.candidate_word = ""
        self.consecutive_frames = 0
        self.last_spoken_word = ""
        self.last_spoken_time = 0.0
        self.no_gesture_count = 0
        self.is_active = True

    def stop(self):
        """Detiene la inferencia en tiempo real."""
        self.is_active = False
        self.sequence = []
        self.candidate_word = ""
        self.consecutive_frames = 0
        self.last_spoken_word = ""

    def process_frame(self, normalized_landmarks, prediction_label_control=None, 
                      progress_bar_control=None, confidence_label_control=None, page_ref=None):
        """
        Procesa cada frame en la ventana deslizante, evalúa el sostenimiento de 10 frames > 85%
        y activa la pronunciación TTS y la actualización de UI sin bloquear.
        """
        if not self.is_active:
            return None, 0.0

        self.sequence.append(normalized_landmarks)
        self.sequence = self.sequence[-30:] # Buffer de 30 frames

        if len(self.sequence) == 30:
            input_tensor = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_tensor, verbose=0)[0]
            predicted_index = int(np.argmax(res))
            confidence = float(res[predicted_index])
            
            self.last_confidence = confidence
            word = self.labels[predicted_index] if predicted_index < len(self.labels) else f"Clase_{predicted_index}"
            page = page_ref if page_ref is not None else self.page

            # 1. Caso de Confianza Alta (> 85%)
            if confidence > self.confidence_threshold:
                self.no_gesture_count = 0

                # Contador consecutivo de la misma palabra
                if word == self.candidate_word:
                    self.consecutive_frames += 1
                else:
                    self.candidate_word = word
                    self.consecutive_frames = 1

                # Verificar si se alcanzó el sostenimiento de 10 frames consecutivos
                if self.consecutive_frames >= self.min_consecutive_frames:
                    self.last_prediction = word

                    # Regla de bloqueo de repetición:
                    # Se pronuncia si es una palabra distinta o si ya pasó tiempo prudencial tras reiniciar
                    now = time.time()
                    should_speak = False
                    if word != self.last_spoken_word:
                        should_speak = True
                    elif now - self.last_spoken_time > 2.5 and self.last_spoken_word == "":
                        should_speak = True

                    if should_speak:
                        self.last_spoken_word = word
                        self.last_spoken_time = now
                        speak_word_offline(word)

                    # Despacho seguro de actualización a la interfaz
                    def _update_confirmed_ui():
                        try:
                            if page and hasattr(page, "is_active") and not page.is_active:
                                return
                            if prediction_label_control:
                                prediction_label_control.value = word.upper()
                                prediction_label_control.update()
                            if progress_bar_control:
                                progress_bar_control.value = min(1.0, confidence)
                                progress_bar_control.update()
                            if confidence_label_control:
                                confidence_label_control.value = f"Confianza: {confidence:.1%}"
                                confidence_label_control.update()
                        except RuntimeError as re:
                            if "destroyed session" in str(re).lower():
                                return
                        except Exception:
                            pass

                    try:
                        if page and hasattr(page, "is_active") and not page.is_active:
                            return word, confidence
                        if page and hasattr(page, "run_thread"):
                            page.run_thread(_update_confirmed_ui)
                        else:
                            _update_confirmed_ui()
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower():
                            return word, confidence
                    except Exception:
                        pass

                    return word, confidence

                else:
                    # En proceso de estabilización (frames < 10)
                    def _update_detecting_ui():
                        try:
                            if page and hasattr(page, "is_active") and not page.is_active:
                                return
                            if confidence_label_control:
                                confidence_label_control.value = f"Detectando {word.upper()} ({self.consecutive_frames}/{self.min_consecutive_frames})..."
                                confidence_label_control.update()
                            if progress_bar_control:
                                progress_bar_control.value = confidence * 0.7
                                progress_bar_control.update()
                        except RuntimeError as re:
                            if "destroyed session" in str(re).lower():
                                return
                        except Exception:
                            pass

                    try:
                        if page and hasattr(page, "is_active") and not page.is_active:
                            return None, confidence
                        if page and hasattr(page, "run_thread"):
                            page.run_thread(_update_detecting_ui)
                        else:
                            _update_detecting_ui()
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower():
                            return None, confidence
                    except Exception:
                        pass

                    return None, confidence

            # 2. Caso de Sin Seña o Baja Confianza (<= 85%)
            else:
                self.consecutive_frames = 0
                self.candidate_word = ""
                self.no_gesture_count += 1

                # Cuando el usuario baja los brazos o se detiene (al menos 8 frames)
                if self.no_gesture_count >= 8:
                    self.last_spoken_word = ""  # Libera el candado de repetición

                # Si pasan más de 15 frames sin seña, reiniciar visuales a reposo
                if self.no_gesture_count >= 15:
                    def _update_idle_ui():
                        try:
                            if page and hasattr(page, "is_active") and not page.is_active:
                                return
                            if prediction_label_control and prediction_label_control.value != "Esperando seña...":
                                prediction_label_control.value = "Esperando seña..."
                                prediction_label_control.update()
                            if progress_bar_control:
                                progress_bar_control.value = 0.0
                                progress_bar_control.update()
                            if confidence_label_control:
                                confidence_label_control.value = "Sin seña detectada"
                                confidence_label_control.update()
                        except RuntimeError as re:
                            if "destroyed session" in str(re).lower():
                                return
                        except Exception:
                            pass

                    try:
                        if page and hasattr(page, "is_active") and not page.is_active:
                            return None, confidence
                        if page and hasattr(page, "run_thread"):
                            page.run_thread(_update_idle_ui)
                        else:
                            _update_idle_ui()
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower():
                            return None, confidence
                    except Exception:
                        pass

                return None, confidence

        return None, 0.0
