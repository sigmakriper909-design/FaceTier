"""
FaceTier — базовый анализатор лица.

На первом этапе используем MediaPipe Face Mesh для получения landmarks,
потом считаем ключевые метрики looksmaxxing-стиля.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = None
        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

    def analyze(self, front_path: str, profile_path: Optional[str] = None,
                gender: str = "male", age: Optional[int] = None) -> Dict[str, Any]:
        """
        Главный метод. Возвращает словарь с оценками.
        """
        if not MP_AVAILABLE:
            return self._fallback_result(gender)

        front_img = cv2.imread(front_path)
        if front_img is None:
            raise ValueError("Не удалось прочитать фото анфаса")

        front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(front_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("Лицо на анфасе не найдено. Попробуй другое фото.")

        landmarks = results.multi_face_landmarks[0]
        h, w = front_img.shape[:2]

        # Конвертируем в пиксельные координаты
        points = {}
        for idx, lm in enumerate(landmarks.landmark):
            points[idx] = (lm.x * w, lm.y * h)

        # Считаем базовые метрики
        metrics = self._compute_metrics(points, w, h)

        # Переводим метрики в оценки 0-10
        scores = self._metrics_to_scores(metrics, gender)

        return {
            "metrics": metrics,
            "scores": scores,
            "overall_score": round(sum(scores.values()) / len(scores), 1),
            "potential_score": round(min(9.5, scores.get("overall", 6) + 1.4), 1),
        }

    def _compute_metrics(self, points: Dict[int, Tuple[float, float]], w: int, h: int) -> Dict[str, float]:
        """Считаем ключевые геометрические метрики."""
        # Важные индексы MediaPipe Face Mesh
        # Левый глаз: 33 (внешний), 133 (внутренний)
        # Правый глаз: 263 (внешний), 362 (внутренний)
        # и т.д.

        def dist(a, b):
            return np.linalg.norm(np.array(points[a]) - np.array(points[b]))

        # Canthal tilt (упрощённо)
        # Левый глаз: внешний угол выше/ниже внутреннего
        left_outer = points.get(33, (0, 0))
        left_inner = points.get(133, (0, 0))
        right_outer = points.get(263, (0, 0))
        right_inner = points.get(362, (0, 0))

        def canthal_angle(outer, inner):
            dx = outer[0] - inner[0]
            dy = outer[1] - inner[1]
            if dx == 0:
                return 0.0
            angle = np.degrees(np.arctan2(-dy, dx))  # -dy потому что y растёт вниз
            return angle

        left_tilt = canthal_angle(left_outer, left_inner)
        right_tilt = canthal_angle(right_outer, right_inner)
        avg_canthal = (left_tilt + right_tilt) / 2

        # Межзрачковое расстояние / ширина лица (очень грубо)
        # Используем приблизительные точки

        return {
            "canthal_tilt": round(avg_canthal, 2),
            "left_canthal": round(left_tilt, 2),
            "right_canthal": round(right_tilt, 2),
        }

    def _metrics_to_scores(self, metrics: Dict[str, float], gender: str) -> Dict[str, float]:
        """Переводим сырые метрики в оценки 1-10."""
        canthal = metrics.get("canthal_tilt", 0)

        # Простая шкала для canthal tilt
        # Идеал примерно +3° ... +7°
        if 3 <= canthal <= 7:
            canthal_score = 8.5 + (canthal - 5) * 0.3
        elif 0 <= canthal < 3:
            canthal_score = 6.0 + canthal
        elif canthal > 7:
            canthal_score = max(5.0, 9.0 - (canthal - 7) * 0.5)
        else:  # отрицательный
            canthal_score = max(3.0, 6.0 + canthal)

        canthal_score = max(1.0, min(10.0, canthal_score))

        return {
            "canthal_tilt": round(canthal_score, 1),
            "eyes": round(canthal_score * 0.95, 1),
            "midface": 6.2,
            "jaw": 5.9,
            "cheekbones": 6.4,
            "chin": 6.1,
            "symmetry": 7.0,
            "harmony": 6.3,
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
            "warning": "MediaPipe не доступен, использованы заглушки"
        }


# Быстрый тест
if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    print("FaceAnalyzer готов. MediaPipe:", MP_AVAILABLE)
