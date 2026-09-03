import os
import json
import time
import threading
import numpy as np
import tensorflow as tf
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

class LSPTesterService:
    """
    Servicio Central de Pruebas e Inferencia en Tiempo Real para el Traductor LSP.
    - Carga de forma desacoplada y segura el modelo binario ('model.keras') y las etiquetas ('labels.json').
    - Estándar estricto bajo el directorio unificado 'data/modelos/'.
    - Ventana deslizante (30 frames) con normalización continua.
    - Confirmación sostenida por al menos 10 frames consecutivos con confianza > 85%.
    - Pronunciación offline por voz con pyttsx3.
    """
    def __init__(self, model_base_dir="data/modelos", model_or_path=None, labels=None, page_ref=None):
        self.model_base_dir = model_base_dir
        self.model = None
        self.labels = {}
        self.page = page_ref
        self.model_path = ""

        # Secuencia y estados de inferencia
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

        # Inicialización directa si se proporcionan parámetros
        if model_or_path is not None:
            if isinstance(model_or_path, tf.keras.Model):
                self.model = model_or_path
                self.model_path = getattr(model_or_path, "name", "loaded_keras_model")
                if labels is not None:
                    self._set_labels(labels)
            elif isinstance(model_or_path, str):
                if os.path.isdir(model_or_path) or (not model_or_path.endswith('.keras') and not model_or_path.endswith('.h5')):
                    self.load_trained_model(model_or_path)
                else:
                    self.model = tf.keras.models.load_model(model_or_path)
                    self.model_path = model_or_path
                    if labels is not None:
                        self._set_labels(labels)

    def load_trained_model(self, category_name):
        """
        Carga el modelo binario y su mapeo de etiquetas de texto.
        category_name puede ser el nombre de la categoría (ej: 'numeros') o una ruta completa.
        """
        if os.path.isdir(category_name):
            category_dir = category_name
        else:
            category_dir = os.path.join(self.model_base_dir, category_name.lower().strip())

        model_path = os.path.join(category_dir, "model.keras")
        labels_path = os.path.join(category_dir, "labels.json")

        # 1. Validaciones físicas previas (Evita colisiones de lectura)
        if not os.path.exists(model_path):
            # Fallback por si existe un modelo antiguo .h5 en la transición
            alt_path = os.path.join(category_dir, "model.h5")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"No se encontró el modelo 'model.keras' en {category_dir}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"No se encontró el archivo de etiquetas 'labels.json' en {category_dir}")

        # 2. Carga del Modelo Binario (EXCLUSIVAMENTE con la API de Keras)
        try:
            # Forzar la carga limpia del archivo binario comprimido
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print(f"[TESTER] Modelo binario importado de forma segura desde: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error al abrir los pesos del modelo con Keras: {str(e)}")

        # 3. Carga del archivo de Etiquetas de Texto (Utilizando UTF-8 limpio)
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                raw_labels = json.load(f)
            self._set_labels(raw_labels)
            print(f"[TESTER] Archivo de mapeo de clases decodificado: {self.labels}")
        except Exception as e:
            raise RuntimeError(f"Error al decodificar etiquetas JSON: {str(e)}")

        return True

    def _set_labels(self, raw_labels):
        """Normaliza las etiquetas para permitir indexación por entero o clave de cadena."""
        if isinstance(raw_labels, dict):
            self.labels = raw_labels
            self.labels_list = [raw_labels[str(i)] if str(i) in raw_labels else raw_labels.get(i, f"Clase_{i}") for i in range(len(raw_labels))]
        elif isinstance(raw_labels, (list, tuple)):
            self.labels = {str(i): w for i, w in enumerate(raw_labels)}
            self.labels_list = list(raw_labels)
        else:
            self.labels = {}
            self.labels_list = []

    def get_word_by_index(self, index: int) -> str:
        """Retorna el nombre de la palabra para un índice de clase dado."""
        if hasattr(self, "labels_list") and 0 <= index < len(self.labels_list):
            return self.labels_list[index]
        if str(index) in self.labels:
            return self.labels[str(index)]
        if index in self.labels:
            return self.labels[index]
        return f"Clase_{index}"

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
        y activa la pronunciación TTS y la actualización de UI sin bloquear ni crashear.
        """
        if not self.is_active or self.model is None:
            return None, 0.0

        self.sequence.append(normalized_landmarks)
        self.sequence = self.sequence[-30:]  # Buffer de 30 frames

        if len(self.sequence) == 30:
            input_tensor = np.expand_dims(self.sequence, axis=0)
            res = self.model.predict(input_tensor, verbose=0)[0]
            predicted_index = int(np.argmax(res))
            confidence = float(res[predicted_index])
            
            self.last_confidence = confidence
            word = self.get_word_by_index(predicted_index)
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

# Alias de compatibilidad y desacoplamiento
LiveTester = LSPTesterService

def cargar_modelo_de_pruebas(ruta_categoria):
    """Función de conveniencia para cargar el modelo de pruebas desacoplado."""
    service = LSPTesterService()
    service.load_trained_model(ruta_categoria)
    return service.model, service.labels
