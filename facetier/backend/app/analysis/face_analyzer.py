"""
FaceTier — анализатор лица на MediaPipe Face Mesh.
Считает canthal tilt, midface, симметрию и другие метрики в стиле looksmaxxing.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import cv2
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = None
        if MP_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def analyze(
        self,
        front_path: str,
        profile_path: Optional[str] = None,
        gender: str = "male",
        age: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not MP_AVAILABLE:
            return self._fallback_result(gender)

        front_img = cv2.imread(front_path)
        if front_img is None:
            raise ValueError("Не удалось прочитать фото анфаса")

        h, w = front_img.shape[:2]
        rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError("Лицо на анфасе не найдено. Сфотографируй лицо прямо и при хорошем освещении.")

        landmarks = results.multi_face_landmarks[0]
        points = {i: (lm.x * w, lm.y * h) for i, lm in enumerate(landmarks.landmark)}

        metrics = self._compute_metrics(points, w, h)
        scores = self._metrics_to_scores(metrics, gender)

        overall = round(float(np.mean(list(scores.values()))), 1)
        potential = round(min(9.5, overall + 1.3), 1)

        return {
            "metrics": metrics,
            "scores": scores,
            "overall_score": overall,
            "potential_score": potential,
        }

    def _dist(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _compute_metrics(self, points: Dict[int, Tuple[float, float]], w: int, h: int) -> Dict[str, float]:
        left_outer = points.get(33, (0, 0))
        left_inner = points.get(133, (0, 0))
        right_outer = points.get(263, (0, 0))
        right_inner = points.get(362, (0, 0))

        def canthal_angle(outer, inner):
            dx = outer[0] - inner[0]
            dy = outer[1] - inner[1]
            if abs(dx) < 1e-6:
                return 0.0
            return float(np.degrees(np.arctan2(-dy, dx)))

        left_tilt = canthal_angle(left_outer, left_inner)
        right_tilt = canthal_angle(right_outer, right_inner)
        avg_canthal = (left_tilt - right_tilt) / 2.0

        left_eye_c = points.get(468, left_inner) if 468 in points else left_inner
        right_eye_c = points.get(473, right_inner) if 473 in points else right_inner
        ipd = self._dist(left_eye_c, right_eye_c)

        top = points.get(10, (w / 2, 0))
        chin = points.get(152, (w / 2, h))
        face_height = max(1.0, abs(chin[1] - top[1]))

        eye_y = (left_inner[1] + right_inner[1]) / 2
        mouth_top = points.get(13, (w / 2, h * 0.7))
        midface_len = abs(mouth_top[1] - eye_y)
        midface_ratio = ipd / max(1.0, midface_len)

        left_jaw = points.get(172, points.get(234, (0, h * 0.7)))
        right_jaw = points.get(397, points.get(454, (w, h * 0.7)))
        jaw_width = self._dist(left_jaw, right_jaw)

        left_cheek = points.get(234, left_jaw)
        right_cheek = points.get(454, right_jaw)
        bizygomatic = self._dist(left_cheek, right_cheek)

        jaw_to_cheek = jaw_width / max(1.0, bizygomatic)

        symmetry_err = abs(left_tilt + right_tilt) + abs(left_outer[1] - right_outer[1]) / h * 100

        return {
            "canthal_tilt": round(avg_canthal, 2),
            "left_canthal": round(left_tilt, 2),
            "right_canthal": round(right_tilt, 2),
            "midface_ratio": round(midface_ratio, 3),
            "jaw_to_cheek": round(jaw_to_cheek, 3),
            "symmetry_error": round(symmetry_err, 2),
            "ipd": round(ipd, 1),
            "face_height": round(face_height, 1),
        }

    def _metrics_to_scores(self, metrics: Dict[str, float], gender: str) -> Dict[str, float]:
        canthal = metrics.get("canthal_tilt", 0.0)
        midface = metrics.get("midface_ratio", 1.0)
        jaw_ratio = metrics.get("jaw_to_cheek", 0.85)
        sym_err = metrics.get("symmetry_error", 5.0)

        if 3 <= canthal <= 7:
            canthal_score = 8.2 + min(1.3, (canthal - 3) * 0.25)
        elif 0 <= canthal < 3:
            canthal_score = 5.5 + canthal * 0.9
        elif canthal > 7:
            canthal_score = max(5.5, 9.0 - (canthal - 7) * 0.4)
        else:
            canthal_score = max(2.5, 5.5 + canthal * 0.7)
        canthal_score = float(np.clip(canthal_score, 1.0, 10.0))

        if 0.95 <= midface <= 1.20:
            midface_score = 7.5 + (midface - 0.95) * 4
        elif midface > 1.20:
            midface_score = max(5.0, 8.5 - (midface - 1.20) * 3)
        else:
            midface_score = max(3.5, 5.0 + midface * 2.5)
        midface_score = float(np.clip(midface_score, 1.0, 10.0))

        ideal_jaw = 0.90 if gender == "male" else 0.82
        jaw_score = 10.0 - abs(jaw_ratio - ideal_jaw) * 18
        jaw_score = float(np.clip(jaw_score, 2.0, 9.5))

        sym_score = 9.5 - sym_err * 0.35
        sym_score = float(np.clip(sym_score, 3.0, 9.8))

        eyes_score = canthal_score * 0.92 + 0.5
        eyes_score = float(np.clip(eyes_score, 1.0, 10.0))

        cheek_score = float(np.clip(6.5 + (jaw_ratio - 0.8) * 5, 3.0, 9.0))
        chin_score = float(np.clip(jaw_score * 0.95, 2.5, 9.2))
        harmony = float(np.mean([canthal_score, midface_score, jaw_score, sym_score]))

        return {
            "canthal_tilt": round(canthal_score, 1),
            "eyes": round(eyes_score, 1),
            "midface": round(midface_score, 1),
            "jaw": round(jaw_score, 1),
            "cheekbones": round(cheek_score, 1),
            "chin": round(chin_score, 1),
            "symmetry": round(sym_score, 1),
            "harmony": round(harmony, 1),
        }

    def _fallback_result(self, gender: str) -> Dict[str, Any]:
        return {
            "metrics": {},
            "scores": {
                "canthal_tilt": 6.5,
                "eyes": 6.5,
                "midface": 6.0,
                "jaw": 5.8,
                "cheekbones": 6.3,
                "chin": 6.0,
                "symmetry": 6.8,
                "harmony": 6.2,
            },
            "overall_score": 6.3,
            "potential_score": 7.7,
            "warning": "MediaPipe недоступен",
        }
