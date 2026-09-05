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
        self.last_raw_frame = None

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
        Genera un lienzo limpio (#F4F8FA) y dibuja un personaje de caricatura infantil amigable inspirado
        en Avatar.png: rostro en tono piel cálido (#FCE2C6), cabello estilizado, ojos grandes azul marino
        con brillo en la esquina superior derecha, cejas dinámicas, boca con apertura bucal en tiempo real,
        camiseta sólida negra con mangas cortas y antebrazos de color piel, y manos tipo guante grueso
        en celeste escolar (#4A90E2, grosor 12) con articulaciones blancas de 8 px.
        Descarta por completo la malla cibernética fría de MediaPipe para evitar fatiga en niños.
        Si no hay detección activa, proyecta un avatar estático amigable por defecto en lugar de una pantalla negra.
        """
        w, h = max(320, image_width), max(240, image_height)
        canvas = np.full((h, w, 3), [250, 248, 244], dtype=np.uint8)

        # Paleta pedagógica inclusiva (Estilo Caricatura Avatar.png)
        color_piel = (198, 226, 252)         # #FCE2C6 en BGR (tono piel cálido)
        color_piel_borde = (165, 195, 230)   # Borde suave para definición anatómica
        color_cabello = (20, 24, 36)         # Marrón oscuro / negro suave
        color_camiseta = (27, 24, 24)        # #18181B en BGR (camiseta negra sólida)
        color_celeste = (226, 144, 74)       # #4A90E2 en BGR (celeste escolar pedagógico)
        color_azul_marino = (93, 54, 26)     # #1A365D en BGR (azul marino institucional)
        color_blanco = (255, 255, 255)
        color_rojo_labios = (92, 92, 226)    # #E25C5C en BGR (rojo amigable para labios)
        color_boca_abierta = (25, 20, 30)    # Cavidad bucal oscura en gesticulación

        FINGER_CONNECTIONS = [
            (1, 2), (2, 3), (3, 4),        # Pulgar
            (5, 6), (6, 7), (7, 8),        # Índice
            (9, 10), (10, 11), (11, 12),   # Medio
            (13, 14), (14, 15), (15, 16), # Anular
            (17, 18), (18, 19), (19, 20)  # Meñique
        ]
        PALM_INDICES = [0, 1, 2, 5, 9, 13, 17]

        try:
            # Reconstruir landmarks desde holistic_results si no se suministró el diccionario
            if (landmarks is None or not any(k in landmarks for k in ["pose", "face", "left_hand", "right_hand"])) and holistic_results is not None:
                try:
                    lh = [[lm.x, lm.y, lm.z] for lm in holistic_results.left_hand_landmarks.landmark] if holistic_results.left_hand_landmarks else [[0.0, 0.0, 0.0]] * 21
                    rh = [[lm.x, lm.y, lm.z] for lm in holistic_results.right_hand_landmarks.landmark] if holistic_results.right_hand_landmarks else [[0.0, 0.0, 0.0]] * 21
                    pose = []
                    if holistic_results.pose_landmarks:
                        for idx in [11, 12, 13, 14, 15, 16]:
                            lm = holistic_results.pose_landmarks.landmark[idx]
                            pose.append([lm.x, lm.y, lm.z])
                    else:
                        pose = [[0.0, 0.0, 0.0]] * 6
                    face = []
                    if holistic_results.face_landmarks:
                        face_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                                        70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 33, 133, 159, 362, 263, 386, 4]
                        for idx in face_indices:
                            lm = holistic_results.face_landmarks.landmark[idx]
                            face.append([lm.x, lm.y, lm.z])
                    else:
                        face = [[0.0, 0.0, 0.0]] * 37
                    landmarks = {
                        "left_hand": np.array(lh, dtype=np.float32),
                        "right_hand": np.array(rh, dtype=np.float32),
                        "pose": np.array(pose, dtype=np.float32),
                        "face": np.array(face, dtype=np.float32),
                    }
                except Exception:
                    landmarks = None

            hlm = holistic_results.face_landmarks.landmark if (holistic_results and getattr(holistic_results, "face_landmarks", None)) else None
            has_pose = bool(landmarks and landmarks.get("pose") is not None and not np.all(landmarks["pose"] == 0.0) and len(landmarks["pose"]) >= 6)
            has_face = bool(hlm is not None or (landmarks and landmarks.get("face") is not None and not np.all(landmarks["face"] == 0.0) and len(landmarks["face"]) == 37))

            # =========================================================================
            # MODO 1: AVATAR ESTÁTICO AMIGABLE (PREVENCIÓN DEFINITIVA DE PANTALLA NEGRA)
            # =========================================================================
            if not has_pose and not has_face:
                cx = w // 2
                cy = int(h * 0.36)
                eye_dist = max(38, int(w * 0.08))
                head_rx = int(eye_dist * 1.35)
                head_ry = int(eye_dist * 1.70)
                head_center = (cx, cy)
                sh_dist = int(w * 0.36)

                # 1. Torso y Camiseta Negra conectándose directamente bajo la barbilla
                chin_y = cy + head_ry
                ls_def = (cx - sh_dist // 2, chin_y + 40)
                rs_def = (cx + sh_dist // 2, chin_y + 40)
                shirt_poly = np.array([
                    [cx - 24, chin_y - 2],
                    [ls_def[0] - 12, ls_def[1]],
                    [ls_def[0] - 35, h],
                    [rs_def[0] + 35, h],
                    [rs_def[0] + 12, rs_def[1]],
                    [cx + 24, chin_y - 2]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [shirt_poly], color_camiseta, cv2.LINE_AA)

                # Brazos y Antebrazos en tono piel
                le_def = (ls_def[0] - 25, int(h * 0.74))
                re_def = (rs_def[0] + 25, int(h * 0.74))
                lw_def = (cx - 70, int(h * 0.88))
                rw_def = (cx + 70, int(h * 0.88))
                # Mangas cortas
                sleeve_l = (int(ls_def[0] + 0.45 * (le_def[0] - ls_def[0])), int(ls_def[1] + 0.45 * (le_def[1] - ls_def[1])))
                sleeve_r = (int(rs_def[0] + 0.45 * (re_def[0] - rs_def[0])), int(rs_def[1] + 0.45 * (re_def[1] - rs_def[1])))
                cv2.line(canvas, ls_def, sleeve_l, color_camiseta, 18, cv2.LINE_AA)
                cv2.line(canvas, rs_def, sleeve_r, color_camiseta, 18, cv2.LINE_AA)
                # Antebrazos en color piel
                cv2.line(canvas, sleeve_l, lw_def, color_piel, 14, cv2.LINE_AA)
                cv2.line(canvas, sleeve_l, lw_def, color_piel_borde, 2, cv2.LINE_AA)
                cv2.line(canvas, sleeve_r, rw_def, color_piel, 14, cv2.LINE_AA)
                cv2.line(canvas, sleeve_r, rw_def, color_piel_borde, 2, cv2.LINE_AA)

                # Manos tipo guante en descanso
                for wpt in [lw_def, rw_def]:
                    cv2.circle(canvas, wpt, 14, color_celeste, -1, cv2.LINE_AA)
                    cv2.circle(canvas, wpt, 4, color_blanco, -1, cv2.LINE_AA)
                    cv2.circle(canvas, wpt, 4, color_celeste, 1, cv2.LINE_AA)

                # 2. Orejas y Cabeza de Color Piel (#FCE2C6) (Rostro limpio sin mejillas ni cuello largo)
                cv2.circle(canvas, (cx - head_rx + 2, cy + 2), 12, color_piel, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx - head_rx + 2, cy + 2), 12, color_piel_borde, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx + head_rx - 2, cy + 2), 12, color_piel, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx + head_rx - 2, cy + 2), 12, color_piel_borde, 2, cv2.LINE_AA)

                cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, color_piel, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, color_piel_borde, 2, cv2.LINE_AA)

                # 3. Cabello Completo, Natural y Estilizado (Polígono cerrado en color oscuro estilo Avatar.png)
                vol = 35
                top_y = cy - head_ry
                hair_pts = [
                    # Límite de frente y cejas
                    [cx - int(head_rx * 0.95), cy - int(head_ry * 0.15)],
                    [cx - int(head_rx * 0.80), cy - int(head_ry * 0.38)],
                    [cx - int(head_rx * 0.50), cy - int(head_ry * 0.52)],
                    [cx - int(head_rx * 0.25), cy - int(head_ry * 0.60)],
                    [cx, cy - int(head_ry * 0.55)],
                    [cx + int(head_rx * 0.25), cy - int(head_ry * 0.60)],
                    [cx + int(head_rx * 0.50), cy - int(head_ry * 0.52)],
                    [cx + int(head_rx * 0.80), cy - int(head_ry * 0.38)],
                    [cx + int(head_rx * 0.95), cy - int(head_ry * 0.15)],
                    # Cúpula de volumen superior sobre el cráneo (+35px)
                    [cx + int(head_rx * 1.05), cy - int(head_ry * 0.35)],
                    [cx + int(head_rx * 0.95), top_y],
                    [cx + int(head_rx * 0.65), top_y - int(vol * 0.75)],
                    [cx + int(head_rx * 0.30), top_y - int(vol * 0.95)],
                    [cx, top_y - vol],
                    [cx - int(head_rx * 0.30), top_y - int(vol * 0.95)],
                    [cx - int(head_rx * 0.65), top_y - int(vol * 0.75)],
                    [cx - int(head_rx * 0.95), top_y],
                    [cx - int(head_rx * 1.05), cy - int(head_ry * 0.35)]
                ]
                hair_poly = np.array(hair_pts, dtype=np.int32)
                cv2.fillPoly(canvas, [hair_poly], color_cabello, cv2.LINE_AA)
                cv2.polylines(canvas, [hair_poly], True, (10, 12, 20), 2, cv2.LINE_AA)

                # 4. Ojos Grandes Azul Marino con Brillo Blanco
                lec = (cx - eye_dist // 2, cy - 4)
                rec = (cx + eye_dist // 2, cy - 4)
                for ec in [lec, rec]:
                    cv2.ellipse(canvas, ec, (14, 11), 0, 0, 360, color_blanco, -1, cv2.LINE_AA)
                    cv2.circle(canvas, ec, 9, color_azul_marino, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (ec[0] + 3, ec[1] - 3), 3, color_blanco, -1, cv2.LINE_AA)

                # Cejas
                cv2.line(canvas, (lec[0] - 14, lec[1] - 12), (lec[0] + 12, lec[1] - 14), color_cabello, 5, cv2.LINE_AA)
                cv2.line(canvas, (rec[0] - 12, rec[1] - 14), (rec[0] + 14, rec[1] - 12), color_cabello, 5, cv2.LINE_AA)

                # Nariz limpia
                cv2.circle(canvas, (cx, cy + 12), 3, color_piel_borde, -1, cv2.LINE_AA)

                # Sonrisa amigable
                smile_pts = np.array([
                    [cx - 20, cy + 28],
                    [cx - 10, cy + 38],
                    [cx + 10, cy + 38],
                    [cx + 20, cy + 28],
                    [cx + 10, cy + 32],
                    [cx - 10, cy + 32]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [smile_pts], (175, 164, 253), cv2.LINE_AA)
                cv2.polylines(canvas, [smile_pts], True, color_rojo_labios, 3, cv2.LINE_AA)

                return canvas

            # =========================================================================
            # MODO 2: AVATAR DINÁMICO EN TIEMPO REAL (SEGUIMIENTO DEL DOCENTE)
            # =========================================================================

            # --- A. Torso y Camiseta Escolar (Estilo Avatar.png) ---
            pose = landmarks.get("pose") if landmarks else None

            if has_pose:
                ls = (int(pose[0][0] * w), int(pose[0][1] * h))
                rs = (int(pose[1][0] * w), int(pose[1][1] * h))
                le = (int(pose[2][0] * w), int(pose[2][1] * h))
                re = (int(pose[3][0] * w), int(pose[3][1] * h))
                lw = (int(pose[4][0] * w), int(pose[4][1] * h))
                rw = (int(pose[5][0] * w), int(pose[5][1] * h))
                sh_dist = max(50, int(np.linalg.norm(np.array(ls) - np.array(rs))))
                neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            else:
                sh_dist = int(w * 0.35)
                cx = w // 2
                neck = (cx, int(h * 0.52))
                ls = (cx - sh_dist // 2, int(h * 0.54))
                rs = (cx + sh_dist // 2, int(h * 0.54))
                le = (ls[0] - 25, int(h * 0.74))
                re = (rs[0] + 25, int(h * 0.74))
                lw = (ls[0] - 10, int(h * 0.88))
                rw = (rs[0] + 10, int(h * 0.88))

            # --- A. Torso y Camiseta Escolar (Estilo Avatar.png con cuello acortado) ---
            pose = landmarks.get("pose") if landmarks else None

            if has_pose:
                ls = (int(pose[0][0] * w), int(pose[0][1] * h))
                rs = (int(pose[1][0] * w), int(pose[1][1] * h))
                le = (int(pose[2][0] * w), int(pose[2][1] * h))
                re = (int(pose[3][0] * w), int(pose[3][1] * h))
                lw = (int(pose[4][0] * w), int(pose[4][1] * h))
                rw = (int(pose[5][0] * w), int(pose[5][1] * h))
                sh_dist = max(50, int(np.linalg.norm(np.array(ls) - np.array(rs))))
            else:
                sh_dist = int(w * 0.35)
                cx = w // 2
                ls = (cx - sh_dist // 2, int(h * 0.54))
                rs = (cx + sh_dist // 2, int(h * 0.54))
                le = (ls[0] - 25, int(h * 0.74))
                re = (rs[0] + 25, int(h * 0.74))
                lw = (ls[0] - 10, int(h * 0.88))
                rw = (rs[0] + 10, int(h * 0.88))

            # Obtener barbilla (landmark 152) para anclar la camiseta directamente
            FACEMESH_CONTOUR = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                54, 103, 67, 109
            ]

            face_pts = None
            if hlm is not None:
                face_pts = np.array([[int(hlm[idx].x * w), int(hlm[idx].y * h)] for idx in FACEMESH_CONTOUR], dtype=np.int32)
                head_center = (int(np.mean(face_pts[:, 0])), int(np.mean(face_pts[:, 1])))
                chin_pt = (int(hlm[152].x * w), int(hlm[152].y * h))
                left_eye_center = (int(hlm[159].x * w), int(hlm[159].y * h))
                right_eye_center = (int(hlm[386].x * w), int(hlm[386].y * h))
                eye_dist = max(20, int(np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))))
                chin_x, chin_y = chin_pt[0], chin_pt[1]
            else:
                face = landmarks.get("face") if landmarks else None
                if face is not None and not np.all(face == 0.0) and len(face) >= 37:
                    nose_pt = (int(face[36][0] * w), int(face[36][1] * h))
                    left_eye_center = (int(np.mean(face[30:33, 0]) * w), int(np.mean(face[30:33, 1]) * h))
                    right_eye_center = (int(np.mean(face[33:36, 0]) * w), int(np.mean(face[33:36, 1]) * h))
                    eye_dist = max(20, int(np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))))
                    head_center = (nose_pt[0], nose_pt[1] - max(8, int(eye_dist * 0.18)))
                    head_rx = max(44, int(eye_dist * 1.35))
                    head_ry = max(56, int(eye_dist * 1.70))
                    chin_x, chin_y = head_center[0], head_center[1] + head_ry
                else:
                    chin_x, chin_y = w // 2, int(h * 0.46)
                    head_center = (chin_x, int(h * 0.36))
                    eye_dist = 40
                    left_eye_center = (chin_x - 20, head_center[1] - 4)
                    right_eye_center = (chin_x + 20, head_center[1] - 4)
                    head_rx, head_ry = 50, 65

            # Polígono trapezoidal sólido de la camiseta negra conectándose DIRECTAMENTE bajo la barbilla (landmark 152)
            neck_hw = max(18, int(sh_dist * 0.16))
            shirt_top_y = chin_y - 2
            shirt_poly = np.array([
                [chin_x - neck_hw, shirt_top_y],
                [ls[0] - 15, ls[1]],
                [ls[0] - 40, h],
                [rs[0] + 40, h],
                [rs[0] + 15, rs[1]],
                [chin_x + neck_hw, shirt_top_y]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [shirt_poly], color_camiseta, cv2.LINE_AA)

            # Mangas cortas de la camiseta y antebrazos en color piel
            sleeve_l_end = (int(ls[0] + 0.45 * (le[0] - ls[0])), int(ls[1] + 0.45 * (le[1] - ls[1])))
            sleeve_r_end = (int(rs[0] + 0.45 * (re[0] - rs[0])), int(rs[1] + 0.45 * (re[1] - rs[1])))
            cv2.line(canvas, ls, sleeve_l_end, color_camiseta, 18, cv2.LINE_AA)
            cv2.line(canvas, rs, sleeve_r_end, color_camiseta, 18, cv2.LINE_AA)

            cv2.line(canvas, sleeve_l_end, lw, color_piel, 14, cv2.LINE_AA)
            cv2.line(canvas, sleeve_l_end, lw, color_piel_borde, 2, cv2.LINE_AA)
            cv2.line(canvas, sleeve_r_end, rw, color_piel, 14, cv2.LINE_AA)
            cv2.line(canvas, sleeve_r_end, rw, color_piel_borde, 2, cv2.LINE_AA)

            # --- B. Rostro y Cabello (Inspirado en Avatar.png, cuello acortado estilizado) ---
            if hlm is not None:
                # Rostro relleno en color piel cálido (#FCE2C6) conectando directamente sobre la camiseta
                cv2.fillPoly(canvas, [face_pts], color_piel, cv2.LINE_AA)
                cv2.polylines(canvas, [face_pts], True, color_piel_borde, 2, cv2.LINE_AA)

                # Cabello Completo, Natural y Estilizado (Polígono cerrado en color oscuro estilo Avatar.png)
                # Basado en landmarks 109, 67, 103 (izq), 336, 296, 334 (der) y landmark 10 con offset de volumen (30-40px)
                top_pt = (int(hlm[10].x * w), int(hlm[10].y * h))
                vol_offset = max(28, min(42, int(eye_dist * 0.75)))

                hair_pts = [
                    # Borde temporal / oreja izquierda
                    [int(hlm[162].x * w), int(hlm[162].y * h)],
                    [int(hlm[21].x * w), int(hlm[21].y * h)],
                    [int(hlm[54].x * w), int(hlm[54].y * h)],
                    # Límite superior de cejas / frente izquierda (landmarks 103, 67, 109)
                    [int(hlm[103].x * w), int(hlm[103].y * h) - 2],
                    [int(hlm[67].x * w), int(hlm[67].y * h) - 4],
                    [int(hlm[109].x * w), int(hlm[109].y * h) - 4],
                    # Centro de la frente
                    [top_pt[0], top_pt[1] + 4],
                    # Límite superior de cejas / frente derecha (landmarks 336, 296, 334)
                    [int(hlm[336].x * w), int(hlm[336].y * h) - 4],
                    [int(hlm[296].x * w), int(hlm[296].y * h) - 4],
                    [int(hlm[334].x * w), int(hlm[334].y * h) - 2],
                    # Borde temporal / oreja derecha
                    [int(hlm[284].x * w), int(hlm[284].y * h)],
                    [int(hlm[251].x * w), int(hlm[251].y * h)],
                    [int(hlm[389].x * w), int(hlm[389].y * h)],
                    # Cúpula externa con volumen superior (30-40px sobre landmark 10)
                    [int(hlm[389].x * w) + 6, int(hlm[389].y * h) - 6],
                    [int(hlm[251].x * w) + 8, int(hlm[251].y * h) - int(vol_offset * 0.35)],
                    [int(hlm[284].x * w) + 8, int(hlm[284].y * h) - int(vol_offset * 0.65)],
                    [int(hlm[334].x * w) + 6, int(hlm[334].y * h) - int(vol_offset * 0.85)],
                    [int(hlm[296].x * w) + 4, int(hlm[296].y * h) - vol_offset],
                    [int(hlm[336].x * w) + 2, int(hlm[336].y * h) - vol_offset],
                    [top_pt[0], top_pt[1] - vol_offset - 3],
                    [int(hlm[109].x * w) - 2, int(hlm[109].y * h) - vol_offset],
                    [int(hlm[67].x * w) - 4, int(hlm[67].y * h) - vol_offset],
                    [int(hlm[103].x * w) - 6, int(hlm[103].y * h) - int(vol_offset * 0.85)],
                    [int(hlm[54].x * w) - 8, int(hlm[54].y * h) - int(vol_offset * 0.65)],
                    [int(hlm[21].x * w) - 8, int(hlm[21].y * h) - int(vol_offset * 0.35)],
                    [int(hlm[162].x * w) - 6, int(hlm[162].y * h) - 6]
                ]
                hair_poly = np.array(hair_pts, dtype=np.int32)
                cv2.fillPoly(canvas, [hair_poly], color_cabello, cv2.LINE_AA)
                cv2.polylines(canvas, [hair_poly], True, (10, 12, 20), 2, cv2.LINE_AA)

            else:
                # Orejas
                ear_r = max(9, int(head_ry * 0.22))
                cv2.circle(canvas, (head_center[0] - head_rx + 2, head_center[1] + 2), ear_r, color_piel, -1, cv2.LINE_AA)
                cv2.circle(canvas, (head_center[0] - head_rx + 2, head_center[1] + 2), ear_r, color_piel_borde, 2, cv2.LINE_AA)
                cv2.circle(canvas, (head_center[0] + head_rx - 2, head_center[1] + 2), ear_r, color_piel, -1, cv2.LINE_AA)
                cv2.circle(canvas, (head_center[0] + head_rx - 2, head_center[1] + 2), ear_r, color_piel_borde, 2, cv2.LINE_AA)

                # Rostro
                cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, color_piel, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, head_center, (head_rx, head_ry), 0, 0, 360, color_piel_borde, 2, cv2.LINE_AA)

                # Cabello Completo, Natural y Estilizado (Polígono cerrado en color oscuro estilo Avatar.png)
                top_x, top_y = head_center[0], head_center[1] - head_ry
                vol = 35
                hair_pts = [
                    [head_center[0] - int(head_rx * 0.95), head_center[1] - int(head_ry * 0.15)],
                    [head_center[0] - int(head_rx * 0.80), head_center[1] - int(head_ry * 0.38)],
                    [head_center[0] - int(head_rx * 0.50), head_center[1] - int(head_ry * 0.52)],
                    [head_center[0] - int(head_rx * 0.25), head_center[1] - int(head_ry * 0.60)],
                    [head_center[0], head_center[1] - int(head_ry * 0.55)],
                    [head_center[0] + int(head_rx * 0.25), head_center[1] - int(head_ry * 0.60)],
                    [head_center[0] + int(head_rx * 0.50), head_center[1] - int(head_ry * 0.52)],
                    [head_center[0] + int(head_rx * 0.80), head_center[1] - int(head_ry * 0.38)],
                    [head_center[0] + int(head_rx * 0.95), head_center[1] - int(head_ry * 0.15)],
                    # Cúpula de volumen superior
                    [head_center[0] + int(head_rx * 1.05), head_center[1] - int(head_ry * 0.35)],
                    [head_center[0] + int(head_rx * 0.95), top_y],
                    [head_center[0] + int(head_rx * 0.65), top_y - int(vol * 0.75)],
                    [head_center[0] + int(head_rx * 0.30), top_y - int(vol * 0.95)],
                    [head_center[0], top_y - vol],
                    [head_center[0] - int(head_rx * 0.30), top_y - int(vol * 0.95)],
                    [head_center[0] - int(head_rx * 0.65), top_y - int(vol * 0.75)],
                    [head_center[0] - int(head_rx * 0.95), top_y],
                    [head_center[0] - int(head_rx * 1.05), head_center[1] - int(head_ry * 0.35)]
                ]
                hair_poly = np.array(hair_pts, dtype=np.int32)
                cv2.fillPoly(canvas, [hair_poly], color_cabello, cv2.LINE_AA)
                cv2.polylines(canvas, [hair_poly], True, (10, 12, 20), 2, cv2.LINE_AA)

            # --- C. Animación Facial Dinámica (Cejas, Ojos con Brillo, Boca con Apertura) ---

            # 1. Cejas dinámicas
            if hlm is not None:
                leb_idx = [70, 63, 105, 66, 107]
                reb_idx = [336, 296, 334, 293, 300]
                leb_pts = np.array([[int(hlm[i].x * w), int(hlm[i].y * h)] for i in leb_idx], dtype=np.int32)
                reb_pts = np.array([[int(hlm[i].x * w), int(hlm[i].y * h)] for i in reb_idx], dtype=np.int32)
                cv2.polylines(canvas, [leb_pts], False, (15, 18, 28), 3, cv2.LINE_AA)
                cv2.polylines(canvas, [reb_pts], False, (15, 18, 28), 3, cv2.LINE_AA)
            elif landmarks and landmarks.get("face") is not None:
                face = landmarks["face"]
                leb_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(20, 25)], dtype=np.int32)
                reb_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(25, 30)], dtype=np.int32)
                cv2.polylines(canvas, [leb_pts], False, (15, 18, 28), 3, cv2.LINE_AA)
                cv2.polylines(canvas, [reb_pts], False, (15, 18, 28), 3, cv2.LINE_AA)

            # 2. Ojos grandes azul marino con brillo animado en esquina superior derecha
            eye_rx = max(11, min(16, int(eye_dist * 0.24)))
            eye_ry = max(9, min(13, int(eye_dist * 0.19)))
            pupil_r = max(6, min(10, int(eye_rx * 0.70)))
            shine_r = max(2, pupil_r // 3)
            for ec in [left_eye_center, right_eye_center]:
                cv2.ellipse(canvas, ec, (eye_rx, eye_ry), 0, 0, 360, color_blanco, -1, cv2.LINE_AA)
                cv2.circle(canvas, ec, pupil_r, color_azul_marino, -1, cv2.LINE_AA)
                # Brillo blanco en esquina superior derecha
                cv2.circle(canvas, (ec[0] + pupil_r // 3, ec[1] - pupil_r // 3), shine_r, color_blanco, -1, cv2.LINE_AA)

            # 4. Boca dinámica con detección de apertura bucal en tiempo real
            if hlm is not None:
                OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
                outer_lip_pts = np.array([[int(hlm[idx].x * w), int(hlm[idx].y * h)] for idx in OUTER_LIP_INDICES], dtype=np.int32)
                cv2.fillPoly(canvas, [outer_lip_pts], color_rojo_labios, cv2.LINE_AA)
                cv2.polylines(canvas, [outer_lip_pts], True, (60, 60, 200), 2, cv2.LINE_AA)

                # Detección de apertura bucal (distancia entre labio superior interno 13 y labio inferior interno 14)
                pt_13 = np.array([hlm[13].x * w, hlm[13].y * h])
                pt_14 = np.array([hlm[14].x * w, hlm[14].y * h])
                lip_open_dist = np.linalg.norm(pt_13 - pt_14)

                if lip_open_dist > 4.5:
                    INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
                    inner_lip_pts = np.array([[int(hlm[idx].x * w), int(hlm[idx].y * h)] for idx in INNER_LIP_INDICES], dtype=np.int32)
                    cv2.fillPoly(canvas, [inner_lip_pts], color_boca_abierta, cv2.LINE_AA)
            elif landmarks and landmarks.get("face") is not None:
                face = landmarks["face"]
                outer_lip_pts = np.array([[int(face[i][0] * w), int(face[i][1] * h)] for i in range(20)], dtype=np.int32)
                cv2.fillPoly(canvas, [outer_lip_pts], color_rojo_labios, cv2.LINE_AA)
                cv2.polylines(canvas, [outer_lip_pts], True, (60, 60, 200), 2, cv2.LINE_AA)

            # --- D. Manos Tipo Guante Definidas (Dedos Separados y Legibles) ---
            if landmarks:
                for hand_key in ["left_hand", "right_hand"]:
                    hand = landmarks.get(hand_key)
                    if hand is not None and not np.all(hand == 0.0) and len(hand) == 21:
                        # 1. Contorno sólido de la base de la palma
                        palm_pts = np.array([[int(hand[idx][0] * w), int(hand[idx][1] * h)] for idx in PALM_INDICES], dtype=np.int32)
                        cv2.fillPoly(canvas, [palm_pts], color_celeste, cv2.LINE_AA)

                        # 2. Dedos definidos de caricatura (grosor 5 px para evitar efecto manopla)
                        for s_idx, e_idx in FINGER_CONNECTIONS:
                            p1 = (int(hand[s_idx][0] * w), int(hand[s_idx][1] * h))
                            p2 = (int(hand[e_idx][0] * w), int(hand[e_idx][1] * h))
                            cv2.line(canvas, p1, p2, color_celeste, 5, cv2.LINE_AA)

                        # 3. Nudillos pequeños blancos (radio 3 px) con borde celeste de 1 px en las 21 articulaciones
                        for pt in hand:
                            cpt = (int(pt[0] * w), int(pt[1] * h))
                            cv2.circle(canvas, cpt, 3, color_blanco, -1, cv2.LINE_AA)
                            cv2.circle(canvas, cpt, 3, color_celeste, 1, cv2.LINE_AA)

        except Exception as ex:
            print(f"[AVATAR] Error renderizando avatar pedagógico: {ex}")

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

            # Guardar último frame para capturas y grabación directa
            self.last_raw_frame = frame.copy()

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
