import cv2
import mediapipe as mp
import numpy as np

class LSPNormalizer:
    def __init__(self):
        """
        Normalizador geométrico optimizado para el Traductor de Lengua de Señas Peruana (LSP).
        - model_complexity=0: Inferencia ultra-ligera en CPU local para alcanzar >30 FPS.
        - refine_face_landmarks=False: Desactiva mallas oculares pesadas innecesarias.
        - Malla Facial Minimalista: 37 puntos clave seleccionados de labios, cejas y ojos.
        """
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Índices de Pose (Torso/Brazos - 6 en total):
        # 11: Left Shoulder, 12: Right Shoulder
        # 13: Left Elbow, 14: Right Elbow
        # 15: Left Wrist, 16: Right Wrist
        self.POSE_INDICES = [11, 12, 13, 14, 15, 16]
        
        # FEATURE SET FACIAL MINIMALISTA (37 Puntos Clave):
        # 1. Contorno y comisuras de labios (20 puntos) - Gesticulación oral y rasgos no manuales LSP
        self.LIPS_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        # 2. Cejas (10 puntos) - Curvatura y altura emocional / gramatical
        self.LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336]
        # 3. Ojos (6 puntos) - Extremos interiores, exteriores y apertura
        self.LEFT_EYE_INDICES = [33, 133, 159]
        self.RIGHT_EYE_INDICES = [362, 263, 386]
        # 4. Nariz (1 punto) - Vértice central del rostro para anclaje cefálico
        self.NOSE_INDICES = [4]
        
        self.FACE_INDICES = (
            self.LIPS_INDICES +
            self.LEFT_EYEBROW_INDICES +
            self.RIGHT_EYEBROW_INDICES +
            self.LEFT_EYE_INDICES +
            self.RIGHT_EYE_INDICES +
            self.NOSE_INDICES
        ) # Total = 37 puntos exactos

    def extract_landmarks(self, frame_rgb):
        """
        Procesa un fotograma y extrae las coordenadas crudas de:
        - Mano Izquierda (21 puntos x 3 = 63)
        - Mano Derecha (21 puntos x 3 = 63)
        - Pose/Torso (6 puntos x 3 = 18)
        - Rostro Minimalista (37 puntos x 3 = 111)
        Total: 85 landmarks (255 coordenadas flotantes)
        """
        results = self.holistic.process(frame_rgb)
        self.last_results = results
        
        # 1. Mano Izquierda (21 puntos x 3 coord = 63)
        lh = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                lh.append([lm.x, lm.y, lm.z])
        else:
            lh = [[0.0, 0.0, 0.0]] * 21
            
        # 2. Mano Derecha (21 puntos x 3 coord = 63)
        rh = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                rh.append([lm.x, lm.y, lm.z])
        else:
            rh = [[0.0, 0.0, 0.0]] * 21

        # 3. Pose / Hombros-Codos-Muñecas (6 puntos x 3 coord = 18)
        pose = []
        if results.pose_landmarks:
            for idx in self.POSE_INDICES:
                lm = results.pose_landmarks.landmark[idx]
                pose.append([lm.x, lm.y, lm.z])
        else:
            pose = [[0.0, 0.0, 0.0]] * 6

        # 4. Rostro Minimalista / Rasgos No Manuales (37 puntos x 3 coord = 111)
        face = []
        if results.face_landmarks:
            for idx in self.FACE_INDICES:
                lm = results.face_landmarks.landmark[idx]
                face.append([lm.x, lm.y, lm.z])
        else:
            face = [[0.0, 0.0, 0.0]] * len(self.FACE_INDICES)

        return {
            "left_hand": np.array(lh, dtype=np.float32),
            "right_hand": np.array(rh, dtype=np.float32),
            "pose": np.array(pose, dtype=np.float32),
            "face": np.array(face, dtype=np.float32)
        }

    def normalize_sequence(self, landmarks_dict):
        """
        Normalización de Invarianza Espacial y de Escala (Fórmula de Anclaje en el Cuello):
        P'_k = (P_k - P_cuello) / ||P_cabeza - P_cuello||_2
        
        - Anclaje de Origen (P_cuello): Punto medio entre los hombros como origen local (0,0,0).
        - Factor de Escala: Distancia euclidiana entre el cuello y el centro de la cabeza (nariz/ojos).
        - Manos: Normalización local e independiente relativa a su propia muñeca y longitud de palma.
        """
        pose = landmarks_dict["pose"]
        lh = landmarks_dict["left_hand"]
        rh = landmarks_dict["right_hand"]
        face = landmarks_dict["face"]

        # 1. Definir Punto de Anclaje del Cuello (P_cuello)
        left_shoulder = pose[0]   # Índice 11 MediaPipe
        right_shoulder = pose[1]  # Índice 12 MediaPipe
        
        if not np.all(left_shoulder == 0.0) and not np.all(right_shoulder == 0.0):
            p_cuello = (left_shoulder + right_shoulder) / 2.0
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        elif not np.all(left_shoulder == 0.0):
            p_cuello = left_shoulder.copy()
            shoulder_dist = 0.3
        elif not np.all(right_shoulder == 0.0):
            p_cuello = right_shoulder.copy()
            shoulder_dist = 0.3
        else:
            p_cuello = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            shoulder_dist = 0.3

        # 2. Definir Centro de la Cabeza (P_cabeza)
        # El último elemento de self.FACE_INDICES es el vértice de la nariz (índice 4)
        nose_point = face[-1] if len(face) == 37 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        if not np.all(nose_point == 0.0):
            p_cabeza = nose_point
        else:
            # Estimación anatómica si la nariz no está detectada
            p_cabeza = np.array([p_cuello[0], p_cuello[1] - max(0.2, shoulder_dist * 0.8), p_cuello[2]], dtype=np.float32)

        # 3. Factor de Escala (Invarianza de Profundidad / Distancia del usuario)
        # ||P_cabeza - P_cuello||_2
        scale_factor = float(np.linalg.norm(p_cabeza - p_cuello))
        if scale_factor < 1e-4:
            scale_factor = float(shoulder_dist) if shoulder_dist > 1e-4 else 1.0

        # Función de normalización anatómica para pose y rostro
        def anchor_normalize(points):
            normalized = []
            for pt in points:
                if np.all(pt == 0.0):
                    normalized.append(pt)
                else:
                    norm_pt = (pt - p_cuello) / scale_factor
                    normalized.append(norm_pt)
            return np.array(normalized, dtype=np.float32)

        norm_pose = anchor_normalize(pose)
        norm_face = anchor_normalize(face)

        # 4. Normalización Local de Manos (Respecto a su propia muñeca y escala de palma)
        def normalize_hand(hand_points, pose_wrist):
            if np.all(hand_points == 0.0):
                return hand_points

            # En MediaPipe Hands, el índice 0 es la muñeca
            ref_wrist = hand_points[0] if not np.all(hand_points[0] == 0.0) else pose_wrist
            if np.all(ref_wrist == 0.0):
                ref_wrist = np.mean(hand_points, axis=0)

            # Longitud de palma (muñeca idx 0 -> nudillo dedo medio idx 9)
            palm_len = np.linalg.norm(hand_points[0] - hand_points[9])
            palm_scale = palm_len if palm_len > 1e-4 else scale_factor
            if palm_scale < 1e-4:
                palm_scale = 1.0

            normalized_hand = []
            for pt in hand_points:
                if np.all(pt == 0.0):
                    normalized_hand.append(pt)
                else:
                    norm_pt = (pt - ref_wrist) / palm_scale
                    normalized_hand.append(norm_pt)
            return np.array(normalized_hand, dtype=np.float32)

        # pose[4] = Left Wrist, pose[5] = Right Wrist
        norm_lh = normalize_hand(lh, pose[4])
        norm_rh = normalize_hand(rh, pose[5])

        # Concatenar todos los puntos en un vector plano de 255 características
        flat_vector = np.concatenate([
            norm_lh.flatten(),  # 21 * 3 = 63
            norm_rh.flatten(),  # 21 * 3 = 63
            norm_pose.flatten(),# 6 * 3 = 18
            norm_face.flatten() # 37 * 3 = 111
        ]).astype(np.float32)   # Total = 255 float32

        return flat_vector

    def close(self):
        self.holistic.close()