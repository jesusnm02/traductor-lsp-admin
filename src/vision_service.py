import cv2
import threading
import time
import base64
import numpy as np
from src.normalizer import LSPNormalizer

class LSPVisionService:
    """
    Servicio de Visión Artificial Concurrente y Libre de Deadlocks para Flet.
    - MediaPipe Holistic instanciado UNA SOLA VEZ en el arranque.
    - Sincronización robusta mediante threading.Event y threading.Lock.
    - Comunicación segura con Flet a través de page.run_thread().
    - Inferencia con ventana deslizante (LiveTester) en hilo secundario.
    """
    def __init__(self, normalizer: LSPNormalizer = None, page_ref=None, on_frame_callback=None, on_state_changed=None, recording_callback=None):
        # 1. MediaPipe inicializado exactamente una sola vez (a través de LSPNormalizer)
        self.normalizer = normalizer if normalizer is not None else LSPNormalizer()
        self.page = page_ref
        self.on_frame_callback = on_frame_callback
        self.on_state_changed = on_state_changed
        self.recording_callback = recording_callback
        
        # Referencias de controles de UI para actualización directa
        self.video_image_control = None
        self.live_tester = None
        self.prediction_label_control = None
        
        # Control de concurrencia seguro
        self.cap = None
        self.is_running = threading.Event()
        self.thread = None
        
        # Parámetros dinámicos configurables por el docente
        self.pre_recording_delay = 3.0  # Tiempo de posicionamiento (segundos)
        self.target_frames = 30         # Cantidad de frames por seña
        
        # Máquina de estados con bloqueo seguro (Thread-Safe)
        self.state_lock = threading.Lock()
        self.current_state = "Inactivo"  # Inactivo, Preparacion, Grabacion, Fin
        self.countdown_duration = 3.0
        self.countdown_start_time = 0.0
        self.recording_buffer = []
        self.recording_category = ""
        self.recording_word = ""

        # Métricas y estado de la cámara
        self.is_camera_obstructed = False
        self.last_brightness = 100.0
        self.fps = 0.0

    def update_params(self, delay: float, frames: int):
        """Permite al docente alterar los parámetros de captura en caliente desde Flet."""
        with self.state_lock:
            self.pre_recording_delay = float(delay)
            self.countdown_duration = float(delay)
            self.target_frames = int(frames)

    def start(self, camera_index=0):
        """Inicia el hilo de captura de forma no bloqueante."""
        if self.is_running.is_set():
            return
        self.is_running.set()
        self.thread = threading.Thread(target=self._camera_loop, args=(camera_index,), daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene el hilo y libera la cámara de manera segura sin colgar la UI."""
        self.is_running.clear()
        self.change_state("Inactivo", "Cámara detenida")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def close(self):
        """Libera todos los recursos de hardware y MediaPipe."""
        self.stop()
        if self.normalizer:
            self.normalizer.close()

    def change_state(self, new_state: str, message: str = ""):
        """Cambia el estado de forma segura y despacha la actualización a la UI mediante page.run_thread."""
        with self.state_lock:
            self.current_state = new_state

        if self.on_state_changed:
            if self.page and hasattr(self.page, "run_thread"):
                self.page.run_thread(self.on_state_changed, new_state, message)
            else:
                self.on_state_changed(new_state, message)

    def start_preparation(self, category_name: str, word: str):
        """Dispara la transición a 'Preparacion' (cuenta regresiva de 3 segundos)."""
        if not self.is_running.is_set():
            self.start()

        with self.state_lock:
            self.recording_category = category_name.strip()
            self.recording_word = word.strip()
            self.recording_buffer = []
            self.countdown_start_time = time.time()

        self.change_state("Preparacion", f"Prepárate en 3s para: '{word.upper()}'")

    # Alias compatible con llamadas previas
    def start_recording(self, category_name: str, word: str):
        self.start_preparation(category_name, word)

    def cancel_recording(self):
        """Cancela la captura actual y vuelve a Inactivo."""
        with self.state_lock:
            self.recording_buffer = []
        self.change_state("Inactivo", "Captura cancelada")

    def _draw_skeletons(self, frame, landmarks):
        """Dibuja los esqueletos optimizados en el frame."""
        h, w, _ = frame.shape

        # Manos (Verde)
        for hand in [landmarks["left_hand"], landmarks["right_hand"]]:
            if not np.all(hand == 0.0):
                for pt in hand:
                    cx, cy = int(pt[0] * w), int(pt[1] * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Pose (Rojo)
        pose = landmarks["pose"]
        if not np.all(pose == 0.0):
            for pt in pose:
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Rostro minimalista 37 puntos (Amarillo)
        face = landmarks["face"]
        if not np.all(face == 0.0):
            for pt in face:
                cx, cy = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

    def _draw_countdown_overlay(self, frame, seconds_left):
        """Dibuja el contador visual de 3 segundos en el centro del frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        box_w, box_h = 380, 160
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

        cv2.putText(frame, "PREPARACION", (x1 + 65, y1 + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        num_str = str(seconds_left)
        cv2.putText(frame, num_str, (x1 + 160, y1 + 115),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 255), 4, cv2.LINE_AA)
        
        cv2.putText(frame, f"Sena: {self.recording_word.upper()} - Listo", (x1 + 35, y1 + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_recording_overlay(self, frame, count, total=30):
        """Dibuja la barra de grabación activa."""
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 180), -1)
        cv2.circle(frame, (30, 22), 10, (0, 0, 255), -1)
        cv2.circle(frame, (30, 22), 12, (255, 255, 255), 2)
        cv2.putText(frame, f"GRABANDO: {self.recording_word.upper()}", (55, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{count}/{total} frames", (w - 150, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    def _camera_loop(self, camera_index=0):
        """Bucle de captura no bloqueante en hilo de fondo secundario."""
        # Inicialización segura con DirectShow en Windows
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        target_frame_time = 1.0 / 30.0
        prev_time = time.time()

        while self.is_running.is_set():
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # Detección de Cámara Obstruida
            mean_vals = cv2.mean(frame)
            mean_brightness = (mean_vals[0] + mean_vals[1] + mean_vals[2]) / 3.0
            self.last_brightness = mean_brightness
            self.is_camera_obstructed = bool(mean_brightness < 15.0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extracción y Normalización de Landmarks (MediaPipe Holistic)
            try:
                landmarks = self.normalizer.extract_landmarks(frame_rgb)
                flat_vector = self.normalizer.normalize_sequence(landmarks)
            except Exception as e:
                print(f"Error en landmarks: {e}")
                continue

            # Dibujar esqueletos en pantalla
            self._draw_skeletons(frame, landmarks)

            # Inferencia en tiempo real (LiveTester) si está activo
            if self.live_tester and getattr(self.live_tester, "is_active", False):
                try:
                    word, conf = self.live_tester.process_frame(
                        flat_vector,
                        prediction_label_control=self.prediction_label_control,
                        page_ref=self.page
                    )
                    if word and conf > 0.85:
                        h, w, _ = frame.shape
                        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 120, 0), -1)
                        cv2.putText(frame, f"SENA: {word.upper()} ({conf:.1%})",
                                    (20, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception as ex:
                    print(f"Error LiveTester: {ex}")

            # Máquina de estados
            with self.state_lock:
                state = self.current_state

            if state == "Preparacion":
                elapsed = time.time() - self.countdown_start_time
                remaining = self.pre_recording_delay - elapsed
                seconds_left = max(1, int(np.ceil(remaining)))
                self._draw_countdown_overlay(frame, seconds_left)

                if elapsed >= self.pre_recording_delay:
                    self.change_state("Grabacion", f"¡Grabando '{self.recording_word.upper()}' ({self.target_frames} frames)!")
                    with self.state_lock:
                        self.recording_buffer = []

            elif state == "Grabacion":
                with self.state_lock:
                    self.recording_buffer.append(flat_vector)
                    count = len(self.recording_buffer)
                    finished = (count >= self.target_frames)
                    if finished:
                        sequence_to_save = list(self.recording_buffer)
                        self.recording_buffer = []
                        cat = self.recording_category
                        word = self.recording_word

                self._draw_recording_overlay(frame, count, self.target_frames)

                if finished:
                    self.change_state("Fin", f"Grabación completada para '{word.upper()}'.")
                    if self.recording_callback:
                        if self.page and hasattr(self.page, "run_thread"):
                            self.page.run_thread(self.recording_callback, cat, word, sequence_to_save)
                        else:
                            threading.Thread(target=self.recording_callback, args=(cat, word, sequence_to_save), daemon=True).start()
                    
                    self.change_state("Inactivo", f"Muestra procesada para '{word.upper()}'.")

            # FPS
            now = time.time()
            self.fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now

            # Codificación Base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Despacho seguro de actualización a Flet UI con page.run_thread()
            if self.on_frame_callback:
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(self.on_frame_callback, img_base64, self.is_camera_obstructed)
                else:
                    self.on_frame_callback(img_base64, self.is_camera_obstructed)

            # Control estricto de FPS (~25 FPS) para eliminar el flickering
            time.sleep(0.04)

# Alias de compatibilidad
LSPCameraHandler = LSPVisionService
