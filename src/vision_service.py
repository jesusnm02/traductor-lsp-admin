import cv2
import threading
import time
import base64
import numpy as np
from src.normalizer import LSPNormalizer

# Constantes de la Máquina de Estados
STATE_IDLE = 0         # Estado 0: Inactivo / Esperando comando
STATE_PREPARATION = 1  # Estado 1: Preparación (Temporizador visual de 3 segundos)
STATE_RECORDING = 2    # Estado 2: Grabación de 30 frames
STATE_COMPLETE = 3     # Estado 3: Fin de captura y retorno a IDLE

class LSPVisionService:
    def __init__(self, normalizer: LSPNormalizer = None, frame_callback=None, recording_callback=None, state_callback=None):
        """
        Servicio de visión artificial y captura de cámara optimizado para Windows.
        - DirectShow (CAP_DSHOW) para inicialización instantánea sin latencia.
        - Inferencia ligera con MediaPipe Holistic (model_complexity=0, 37 puntos faciales).
        - Máquina de estados: 0 (Inactivo), 1 (Preparación 3s), 2 (Grabación), 3 (Fin).
        - Concurrencia desacoplada en hilo secundario para tasa fluida de >= 30 FPS.
        - FIX de congelamiento: soporte para actualización directa del control Image de Flet.
        - Soporte para inferencia en tiempo real (LiveTester) mediante ventana deslizante.
        """
        self.normalizer = normalizer if normalizer is not None else LSPNormalizer()
        self.frame_callback = frame_callback
        self.recording_callback = recording_callback
        self.state_callback = state_callback
        
        # Referencias opcionales directas a controles de Flet para refresco ultrarrápido
        self.video_image_control = None
        self.live_tester = None
        self.prediction_label_control = None
        
        self.cap = None
        self.is_running = False
        self.camera_thread = None
        
        # Detección de cámara tapada u obstruida
        self.is_camera_obstructed = False
        self.last_brightness = 100.0
        
        # Métricas de rendimiento
        self.fps = 0.0
        
        # MÁQUINA DE ESTADOS
        self.state = STATE_IDLE
        self.countdown_duration = 3.0  # 3 segundos de preparación
        self.countdown_start_time = 0.0
        self.recording_buffer = []      # Buffer para 30 frames normalizados (255 coords c/u)
        self.recording_category = ""
        self.recording_word = ""

        # Lock de sincronización para concurrencia segura
        self.lock = threading.Lock()

    def start(self, camera_index=0):
        """Inicializa la cámara web usando DirectShow (CAP_DSHOW) en Windows."""
        with self.lock:
            if self.is_running:
                return
            
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.camera_thread.start()

    def stop(self):
        """Detiene el hilo de captura y libera la cámara de manera segura."""
        with self.lock:
            self.is_running = False
            self.state = STATE_IDLE
            self.recording_buffer = []
            
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
            
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def close(self):
        """Libera todos los recursos de hardware (cámara) y modelos (MediaPipe)."""
        self.stop()
        if self.normalizer:
            self.normalizer.close()

    def start_preparation(self, category_name: str, word: str):
        """
        Inicia la Máquina de Estados:
        Transición a Estado 1 (Preparación): Temporizador visual de 3 segundos antes de grabar.
        Si la cámara no está activa, la inicia automáticamente.
        """
        if not self.is_running:
            self.start()

        with self.lock:
            self.recording_category = category_name.strip()
            self.recording_word = word.strip()
            self.recording_buffer = []
            self.countdown_start_time = time.time()
            self.state = STATE_PREPARATION

        if self.state_callback:
            self.state_callback(STATE_PREPARATION, f"Preparación: 3 segundos para '{word.upper()}'")

    # Alias compatible con métodos anteriores
    def start_recording(self, category_name: str, word: str):
        self.start_preparation(category_name, word)

    def cancel_recording(self):
        """Cancela el proceso actual y vuelve a Estado 0 (Inactivo)."""
        with self.lock:
            self.state = STATE_IDLE
            self.recording_buffer = []

        if self.state_callback:
            self.state_callback(STATE_IDLE, "Proceso cancelado.")

    def _draw_skeletons(self, frame, landmarks):
        """Dibuja esqueletos y puntos clave procesados sobre el frame."""
        h, w, _ = frame.shape

        # 1. Manos (Verde)
        for hand in [landmarks["left_hand"], landmarks["right_hand"]]:
            if not np.all(hand == 0.0):
                for pt in hand:
                    cx, cy = int(pt[0] * w), int(pt[1] * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # 2. Pose (Hombros, Codos, Muñecas - Rojo)
        pose = landmarks["pose"]
        if not np.all(pose == 0.0):
            for pt in pose:
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 3. Rostro Minimalista: Labios, Cejas, Ojos, Nariz (Amarillo/Cian)
        face = landmarks["face"]
        if not np.all(face == 0.0):
            for pt in face:
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

    def _draw_countdown_overlay(self, frame, seconds_left):
        """Dibuja el contador visual de preparación (Estado 1) centrado y destacado."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        box_w, box_h = 380, 160
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        
        # Fondo translúcido oscuro
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

        # Título
        cv2.putText(frame, "PREPARACION", (x1 + 65, y1 + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Número grande
        num_str = str(seconds_left)
        cv2.putText(frame, num_str, (x1 + 160, y1 + 115),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
        
        # Subtítulo
        cv2.putText(frame, f"Seña: {self.recording_word.upper()} - Coloquese listo", (x1 + 35, y1 + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_recording_overlay(self, frame, count, total=30):
        """Dibuja la barra de grabación activa (Estado 2)."""
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 180), -1)
        cv2.circle(frame, (30, 22), 10, (0, 0, 255), -1)
        cv2.circle(frame, (30, 22), 12, (255, 255, 255), 2)
        cv2.putText(frame, f"GRABANDO: {self.recording_word.upper()}", (55, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{count}/{total} frames", (w - 150, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    def _capture_loop(self):
        """Bucle principal de procesamiento de frames en hilo secundario dedicado."""
        target_frame_time = 1.0 / 30.0  # Intervalo para 30 FPS
        prev_time = time.time()

        while True:
            loop_start = time.time()
            
            with self.lock:
                if not self.is_running:
                    break
                cap = self.cap
            
            if cap is None or not cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Invertir horizontalmente para efecto espejo natural
            frame = cv2.flip(frame, 1)

            # Detección de Cámara Obstruida / Iluminación insuficiente
            mean_vals = cv2.mean(frame)
            mean_brightness = (mean_vals[0] + mean_vals[1] + mean_vals[2]) / 3.0
            self.last_brightness = mean_brightness
            self.is_camera_obstructed = bool(mean_brightness < 15.0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extraer y normalizar landmarks
            try:
                landmarks = self.normalizer.extract_landmarks(frame_rgb)
                flat_vector = self.normalizer.normalize_sequence(landmarks)
            except Exception as e:
                print(f"Error procesando landmarks: {str(e)}")
                continue

            # Dibujar esqueletos sobre el fotograma
            self._draw_skeletons(frame, landmarks)

            # --- VALIDACIÓN EN VIVO: LIVE TESTER (SLIDING WINDOW) ---
            if self.live_tester and getattr(self.live_tester, "is_active", False):
                try:
                    pred_label, conf = self.live_tester.process_frame(flat_vector, self.prediction_label_control)
                    if pred_label and conf > 0.85:
                        h, w, _ = frame.shape
                        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 120, 0), -1)
                        cv2.putText(frame, f"SENA DETECTADA: {pred_label.upper()} ({conf:.1%})",
                                    (20, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as ex:
                    print(f"Error en LiveTester: {ex}")

            # --- GESTIÓN DE LA MÁQUINA DE ESTADOS ---
            with self.lock:
                current_state = self.state

            # Estado 1: Preparación (Contador de 3 segundos)
            if current_state == STATE_PREPARATION:
                elapsed = time.time() - self.countdown_start_time
                remaining = self.countdown_duration - elapsed
                seconds_display = max(1, int(np.ceil(remaining)))
                
                self._draw_countdown_overlay(frame, seconds_display)

                if elapsed >= self.countdown_duration:
                    with self.lock:
                        self.state = STATE_RECORDING
                        self.recording_buffer = []
                    if self.state_callback:
                        self.state_callback(STATE_RECORDING, f"¡Grabando '{self.recording_word.upper()}' ahora!")

            # Estado 2: Grabación de 30 frames
            elif current_state == STATE_RECORDING:
                with self.lock:
                    self.recording_buffer.append(flat_vector)
                    count = len(self.recording_buffer)
                    finished = (count >= 30)
                    if finished:
                        sequence_to_save = list(self.recording_buffer)
                        self.recording_buffer = []
                        self.state = STATE_COMPLETE
                        cat = self.recording_category
                        word = self.recording_word

                self._draw_recording_overlay(frame, count, 30)

                if finished:
                    if self.recording_callback:
                        threading.Thread(
                            target=self.recording_callback,
                            args=(cat, word, sequence_to_save),
                            daemon=True
                        ).start()
                    
                    with self.lock:
                        self.state = STATE_IDLE
                    if self.state_callback:
                        self.state_callback(STATE_IDLE, f"Grabación completada para '{word.upper()}'.")

            # Cálculo de FPS real
            now = time.time()
            self.fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now

            # Codificación a JPEG y Base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # FIX OBLIGATORIO: Actualización directa y explícita al control Image de Flet
            # Evita la llamada global a page.update() que congelaba la UI
            if self.video_image_control is not None:
                if hasattr(self.video_image_control, "src"):
                    self.video_image_control.src = f"data:image/jpeg;base64,{img_base64}"
                if hasattr(self.video_image_control, "src_base64"):
                    self.video_image_control.src_base64 = img_base64
                try:
                    self.video_image_control.update()
                except Exception:
                    pass

            # Callback general para la UI
            if self.frame_callback:
                try:
                    self.frame_callback(img_base64, self.is_camera_obstructed)
                except TypeError:
                    self.frame_callback(img_base64)
                except Exception:
                    pass

            # Control dinámico de framerate (~30 FPS)
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, target_frame_time - elapsed)
            time.sleep(sleep_time)

# Alias de compatibilidad
LSPCameraHandler = LSPVisionService
