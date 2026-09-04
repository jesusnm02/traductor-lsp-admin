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

        # Modo Avatar de Privacidad (Títere vectorial de-identificado)
        self.privacy_avatar_mode = False
        self.avatar_recording_frames = []

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
            try:
                if self.page and hasattr(self.page, "is_active") and not self.page.is_active:
                    return
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(self.on_state_changed, new_state, message)
                else:
                    self.on_state_changed(new_state, message)
            except RuntimeError as re:
                if "destroyed session" in str(re).lower():
                    return
            except Exception:
                pass

    def start_preparation(self, category_name: str, word: str):
        """Dispara la transición a 'Preparacion' (cuenta regresiva de 3 segundos)."""
        if not self.is_running.is_set():
            self.start()

        with self.state_lock:
            self.recording_category = category_name.strip()
            self.recording_word = word.strip()
            self.recording_buffer = []
            self.avatar_recording_frames = []
            self.countdown_start_time = time.time()

        self.change_state("Preparacion", f"Prepárate en 3s para: '{word.upper()}'")

    # Alias compatible con llamadas previas
    def start_recording(self, category_name: str, word: str):
        self.start_preparation(category_name, word)

    def cancel_recording(self):
        """Cancela la captura actual y vuelve a Inactivo."""
        with self.state_lock:
            self.recording_buffer = []
            self.avatar_recording_frames = []
        self.change_state("Inactivo", "Captura cancelada")

    def set_privacy_avatar_mode(self, enabled: bool):
        """Activa o desactiva el Modo Avatar de Privacidad (descarte de píxeles reales)."""
        with self.state_lock:
            self.privacy_avatar_mode = bool(enabled)
            self.avatar_recording_frames = []

    def render_privacy_avatar(self, image_width: int, image_height: int, landmarks: dict = None, holistic_results=None) -> np.ndarray:
        """
        Genera un lienzo limpio (#F4F8FA) y dibuja un avatar animado (títere vectorial)
        basado en los landmarks de pose, rostro y manos.
        Descarta completamente el feed RGB real para proteger la privacidad del docente.
        """
        w, h = image_width, image_height
        canvas = np.full((h, w, 3), [250, 248, 244], dtype=np.uint8)

        if landmarks is None and holistic_results is None:
            return canvas

        color_celeste = (226, 144, 74)   # #4A90E2 en BGR
        color_azul_oscuro = (93, 54, 26) # #1A365D en BGR
        color_blanco = (255, 255, 255)

        # Distintivo superior de Privacidad
        cv2.putText(canvas, "AVATAR DE PRIVACIDAD (DE-IDENTIFICADO)", (16, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, color_azul_oscuro, 1, cv2.LINE_AA)

        # 1. Pose / Torso y Brazos
        pose = landmarks.get("pose") if landmarks else None
        if pose is not None and not np.all(pose == 0.0) and len(pose) >= 6:
            ls = (int(pose[0][0] * w), int(pose[0][1] * h))
            rs = (int(pose[1][0] * w), int(pose[1][1] * h))
            le = (int(pose[2][0] * w), int(pose[2][1] * h))
            re = (int(pose[3][0] * w), int(pose[3][1] * h))
            lw = (int(pose[4][0] * w), int(pose[4][1] * h))
            rw = (int(pose[5][0] * w), int(pose[5][1] * h))
            neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)

            sh_dist = int(np.linalg.norm(np.array(ls) - np.array(rs)))
            torso_bottom = (neck[0], int(neck[1] + max(60, sh_dist * 0.9)))
            cv2.line(canvas, neck, torso_bottom, color_azul_oscuro, 5, cv2.LINE_AA)
            cv2.line(canvas, ls, rs, color_azul_oscuro, 4, cv2.LINE_AA)

            # Brazos (Azul oscuro exterior + blanco interior estilo vector)
            for p1, p2 in [(ls, le), (le, lw), (rs, re), (re, rw)]:
                cv2.line(canvas, p1, p2, color_azul_oscuro, 5, cv2.LINE_AA)
                cv2.line(canvas, p1, p2, color_blanco, 2, cv2.LINE_AA)

            for pt in [ls, rs, le, re]:
                cv2.circle(canvas, pt, 6, color_celeste, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 2, color_blanco, -1, cv2.LINE_AA)

        # 2. Rostro Minimalista (37 puntos: cejas, ojos, labios, nariz)
        face = landmarks.get("face") if landmarks else None
        if face is not None and not np.all(face == 0.0) and len(face) == 37:
            # Silueta cefálica suave
            nose_pt = (int(face[36][0] * w), int(face[36][1] * h))
            left_eye_center = (int(np.mean(face[30:33, 0]) * w), int(np.mean(face[30:33, 1]) * h))
            right_eye_center = (int(np.mean(face[33:36, 0]) * w), int(np.mean(face[33:36, 1]) * h))
            eye_dist = int(np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center)))
            head_rx = max(35, int(eye_dist * 1.3))
            head_ry = max(45, int(eye_dist * 1.6))
            head_center = (nose_pt[0], nose_pt[1] - 5)

            # Fondo suave de cabeza y borde celeste
            cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, (235, 242, 248), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, color_celeste, 2, cv2.LINE_AA)

            # Cejas (Gesticulación emocional / gramatical)
            leb_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(20, 25)], dtype=np.int32)
            reb_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(25, 30)], dtype=np.int32)
            cv2.polylines(canvas, [leb_pts], False, color_azul_oscuro, 3, cv2.LINE_AA)
            cv2.polylines(canvas, [reb_pts], False, color_azul_oscuro, 3, cv2.LINE_AA)

            # Ojos estilizados
            for ec in [left_eye_center, right_eye_center]:
                cv2.circle(canvas, ec, 5, color_azul_oscuro, -1, cv2.LINE_AA)
                cv2.circle(canvas, (ec[0] - 1, ec[1] - 1), 2, color_blanco, -1, cv2.LINE_AA)

            # Nariz (anclaje central)
            cv2.circle(canvas, nose_pt, 3, color_celeste, -1, cv2.LINE_AA)

            # Labios (Gesticulación oral y rasgos no manuales)
            lip_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(20)], dtype=np.int32)
            cv2.polylines(canvas, [lip_pts], True, color_azul_oscuro, 2, cv2.LINE_AA)
            cv2.polylines(canvas, [lip_pts], True, color_celeste, 1, cv2.LINE_AA)

        # 3. Manos de Alta Fidelidad (21 puntos por mano)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
            (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
            (9, 13), (13, 14), (14, 15), (15, 16), # Anular
            (13, 17), (17, 18), (18, 19), (19, 20),# Meñique
            (0, 17)                                # Palma
        ]
        if landmarks:
            for hand_key in ["left_hand", "right_hand"]:
                hand = landmarks.get(hand_key)
                if hand is not None and not np.all(hand == 0.0) and len(hand) == 21:
                    # Huesos / Conexiones
                    for s_idx, e_idx in HAND_CONNECTIONS:
                        p1 = (int(hand[s_idx][0] * w), int(hand[s_idx][1] * h))
                        p2 = (int(hand[e_idx][0] * w), int(hand[e_idx][1] * h))
                        cv2.line(canvas, p1, p2, color_azul_oscuro, 4, cv2.LINE_AA)
                        cv2.line(canvas, p1, p2, color_blanco, 2, cv2.LINE_AA)

                    # Nudillos y Articulaciones
                    for pt in hand:
                        cpt = (int(pt[0] * w), int(pt[1] * h))
                        cv2.circle(canvas, cpt, 4, color_celeste, -1, cv2.LINE_AA)
                        cv2.circle(canvas, cpt, 1, color_blanco, -1, cv2.LINE_AA)

        return canvas

    def _save_avatar_gif(self, category: str, word: str, frames: list):
        """Guarda un GIF didáctico de-identificado en segundo plano sin bloquear."""
        try:
            import os
            from PIL import Image
            from src.cloud_service import DATA_DIR
            target_dir = os.path.join(DATA_DIR, "muestras", category.lower().strip(), word.lower().strip())
            os.makedirs(target_dir, exist_ok=True)
            gif_path = os.path.join(target_dir, "guia.gif")
            
            pil_images = []
            for f in frames:
                small = cv2.resize(f, (320, 240))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
                
            if pil_images:
                pil_images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    optimize=True,
                    duration=40,
                    loop=0
                )
                print(f"[AVATAR] Guía animada de-identificada guardada en: {gif_path}")
        except Exception as e:
            print(f"[AVATAR] Error guardando guía GIF: {e}")

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

            # Sustitución del feed RGB o dibujo sobre cámara real
            if self.privacy_avatar_mode:
                # El feed original de la cámara se descarta COMPLETAMENTE para proteger la identidad biométrica
                avatar_frame = self.render_privacy_avatar(
                    frame.shape[1],
                    frame.shape[0],
                    landmarks=landmarks,
                    holistic_results=getattr(self.normalizer, 'last_results', None)
                )
                frame = avatar_frame
            else:
                # Modo normal: Dibujar esqueletos sobre la cámara real
                self._draw_skeletons(frame, landmarks)

            # Inferencia en tiempo real (LiveTester) si está activo
            if self.live_tester and getattr(self.live_tester, "is_active", False):
                try:
                    word, conf = self.live_tester.process_frame(
                        flat_vector,
                        prediction_label_control=self.prediction_label_control,
                        progress_bar_control=getattr(self, "progress_bar_control", None),
                        confidence_label_control=getattr(self, "confidence_label_control", None),
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
                        self.avatar_recording_frames = []

            elif state == "Grabacion":
                with self.state_lock:
                    self.recording_buffer.append(flat_vector)
                    if self.privacy_avatar_mode:
                        self.avatar_recording_frames.append(frame.copy())
                    count = len(self.recording_buffer)
                    finished = (count >= self.target_frames)
                    if finished:
                        sequence_to_save = list(self.recording_buffer)
                        self.recording_buffer = []
                        cat = self.recording_category
                        word = self.recording_word
                        avatar_frames_to_save = list(self.avatar_recording_frames)
                        self.avatar_recording_frames = []

                self._draw_recording_overlay(frame, count, self.target_frames)

                if finished:
                    self.change_state("Fin", f"Grabación completada para '{word.upper()}'.")
                    if self.privacy_avatar_mode and avatar_frames_to_save:
                        threading.Thread(
                            target=self._save_avatar_gif,
                            args=(cat, word, avatar_frames_to_save),
                            daemon=True
                        ).start()

                    if self.recording_callback:
                        try:
                            if self.page and hasattr(self.page, "is_active") and not self.page.is_active:
                                self.stop()
                                break
                            if self.page and hasattr(self.page, "run_thread"):
                                self.page.run_thread(self.recording_callback, cat, word, sequence_to_save)
                            else:
                                threading.Thread(target=self.recording_callback, args=(cat, word, sequence_to_save), daemon=True).start()
                        except RuntimeError as re:
                            if "destroyed session" in str(re).lower():
                                self.stop()
                                break
                        except Exception:
                            pass
                    
                    self.change_state("Inactivo", f"Muestra procesada para '{word.upper()}'.")

            # FPS
            now = time.time()
            self.fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now

            # Codificación Base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Despacho seguro de actualización a Flet UI con page.run_thread()
            if self.on_frame_callback and self.is_running.is_set():
                try:
                    if self.page and hasattr(self.page, "is_active") and not self.page.is_active:
                        self.stop()
                        break
                    if self.page and hasattr(self.page, "run_thread"):
                        self.page.run_thread(self.on_frame_callback, img_base64, self.is_camera_obstructed)
                    else:
                        self.on_frame_callback(img_base64, self.is_camera_obstructed)
                except RuntimeError as re:
                    if "destroyed session" in str(re).lower():
                        self.stop()
                        break
                except Exception:
                    pass

            # Control estricto de FPS (~25 FPS) para eliminar el flickering
            time.sleep(0.04)

# Alias de compatibilidad
LSPCameraHandler = LSPVisionService
