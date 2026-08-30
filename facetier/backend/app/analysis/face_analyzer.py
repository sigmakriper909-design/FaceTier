"""
FaceTier — анализатор лица на MediaPipe Face Mesh.
Считает метрики looksmaxxing-стиля и отдаёт нормализованные точки для оверлеев.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

MP_AVAILABLE = False
MP_IMPORT_ERROR = None

try:
    import cv2
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception as e:
    MP_IMPORT_ERROR = f"{type(e).__name__}: {e}"


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
            result = self._fallback_result(gender)
            result["warning"] = f"MediaPipe недоступен: {MP_IMPORT_ERROR}"
            result["mp_available"] = False
            return result

        front_img = cv2.imread(front_path)
        if front_img is None:
            raise ValueError("Не удалось прочитать фото анфаса")

        h, w = front_img.shape[:2]
        rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError("Лицо на анфасе не найдено. Сфотографируй лицо прямо и при хорошем освещении.")

        landmarks = results.multi_face_landmarks[0]
        pts_n = {i: (lm.x, lm.y) for i, lm in enumerate(landmarks.landmark)}
        pts_px = {i: (lm.x * w, lm.y * h) for i, lm in enumerate(landmarks.landmark)}

        metrics = self._compute_metrics(pts_px, w, h)
        scores = self._metrics_to_scores(metrics, gender)
        overlays = self._build_overlays(pts_n)

        overall = round(float(np.mean(list(scores.values()))), 1)
        potential = round(min(9.5, overall + 1.3), 1)

        return {
            "metrics": metrics,
            "scores": scores,
            "overlays": overlays,
            "overall_score": overall,
            "potential_score": potential,
            "mp_available": True,
        }

    def _dist(self, p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _n(self, pts: Dict, idx: int) -> Optional[List[float]]:
        p = pts.get(idx)
        if p is None:
            return None
        return [round(float(p[0]), 4), round(float(p[1]), 4)]

    def _pts(self, pts: Dict, ids: List[int]) -> List[List[float]]:
        out = []
        for i in ids:
            p = self._n(pts, i)
            if p:
                out.append(p)
        return out

    def _compute_metrics(self, pts_px: Dict, w: int, h: int) -> Dict[str, float]:
        """Canthal tilt, midface ratio, jaw-to-cheek, symmetry error."""
        def safe(i):
            return pts_px.get(i)

        canthal = 0.0
        left_o, left_i = safe(33), safe(133)
        right_o, right_i = safe(263), safe(362)
        angles = []
        if left_o and left_i:
            dx = left_i[0] - left_o[0]
            if abs(dx) > 1e-6:
                angles.append(float(np.degrees(np.arctan2(-(left_o[1] - left_i[1]), abs(dx)))))
        if right_o and right_i:
            dx = right_o[0] - right_i[0]
            if abs(dx) > 1e-6:
                angles.append(float(np.degrees(np.arctan2(-(right_o[1] - right_i[1]), abs(dx)))))
        if angles:
            canthal = float(np.mean(angles))

        top = safe(10)
        subnasale = safe(2)
        menton = safe(152)
        midface_ratio = 1.0
        if top and subnasale and menton:
            upper = self._dist(top, subnasale)
            lower = self._dist(subnasale, menton)
            if lower > 1e-6:
                midface_ratio = float(upper / lower)

        left_jaw, right_jaw = safe(172), safe(397)
        left_cheek, right_cheek = safe(234), safe(454)
        jaw_to_cheek = 0.85
        if left_jaw and right_jaw and left_cheek and right_cheek:
            jaw_w = self._dist(left_jaw, right_jaw)
            cheek_w = self._dist(left_cheek, right_cheek)
            if cheek_w > 1e-6:
                jaw_to_cheek = float(jaw_w / cheek_w)

        midline_x = None
        if safe(10) and safe(152):
            midline_x = (safe(10)[0] + safe(152)[0]) / 2.0
        pairs = [(33, 263), (133, 362), (234, 454), (172, 397), (61, 291)]
        errs = []
        if midline_x is not None:
            for li, ri in pairs:
                lp, rp = safe(li), safe(ri)
                if lp and rp:
                    left_off = midline_x - lp[0]
                    right_off = rp[0] - midline_x
                    errs.append(abs(left_off - right_off) / max(w, 1) * 100.0)
        symmetry_error = float(np.mean(errs)) if errs else 5.0

        return {
            "canthal_tilt": round(canthal, 2),
            "midface_ratio": round(midface_ratio, 3),
            "jaw_to_cheek": round(jaw_to_cheek, 3),
            "symmetry_error": round(symmetry_error, 2),
        }

    def _build_overlays(self, pts: Dict) -> Dict[str, Any]:
        eyes_pts = self._pts(pts, [33, 133, 263, 362, 468, 473, 70, 300])
        midface_pts = self._pts(pts, [10, 1, 13, 14, 234, 454, 133, 362])
        jaw_pts = self._pts(pts, [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397])
        cheek_pts = self._pts(pts, [234, 454, 116, 345])
        chin_pts = self._pts(pts, [152, 175, 199, 176, 148, 377, 400])
        face_outline = self._pts(pts, [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ])
        center_line = self._pts(pts, [10, 1, 13, 14, 152])

        return {
            "front": {
                "overall": {"kind": "polyline", "points": face_outline, "color": "#7c5cff"},
                "eyes": {
                    "kind": "points+lines",
                    "points": eyes_pts,
                    "lines": [self._pts(pts, [33, 133]), self._pts(pts, [263, 362])],
                    "color": "#00d4aa",
                },
                "midface": {
                    "kind": "points+lines",
                    "points": midface_pts,
                    "lines": [self._pts(pts, [133, 362]), self._pts(pts, [10, 152])],
                    "color": "#7c5cff",
                },
                "jaw": {"kind": "polyline", "points": jaw_pts, "color": "#ffb020"},
                "cheekbones": {
                    "kind": "points+lines",
                    "points": cheek_pts,
                    "lines": [self._pts(pts, [234, 454])],
                    "color": "#ff5c7a",
                },
                "chin": {"kind": "points", "points": chin_pts, "color": "#00d4aa"},
                "symmetry": {
                    "kind": "points+lines",
                    "points": center_line + eyes_pts[:4],
                    "lines": [center_line, self._pts(pts, [33, 263])],
                    "color": "#7c5cff",
                },
                "harmony": {"kind": "polyline", "points": face_outline, "color": "#00d4aa"},
            },
            "profile": {
                "jaw": {"kind": "note"},
                "chin": {"kind": "note"},
                "midface": {"kind": "note"},
            },
        }

    def _metrics_to_scores(self, metrics: Dict[str, float], gender: str) -> Dict[str, float]:
        canthal = metrics.get("canthal_tilt", 0.0)
        midface = metrics.get("midface_ratio", 1.0)
        jaw_ratio = metrics.get("jaw_to_cheek", 0.85)
        sym_err = metrics.get("symmetry_error", 5.0)

        if 3 <= canthal <= 7:
            canthal_score = 8.0 + min(1.5, (canthal - 3) * 0.3)
        elif 1 <= canthal < 3:
            canthal_score = 6.0 + (canthal - 1) * 1.0
        elif canthal > 7:
            canthal_score = max(5.5, 9.2 - (canthal - 7) * 0.4)
        else:
            canthal_score = max(2.0, 5.5 + canthal * 0.8)
        canthal_score = float(np.clip(canthal_score, 1.0, 10.0))

        if 0.95 <= midface <= 1.25:
            midface_score = 7.2 + (midface - 0.95) * 5
        elif midface > 1.25:
            midface_score = max(4.5, 8.5 - (midface - 1.25) * 4)
        else:
            midface_score = max(3.0, 4.5 + midface * 3)
        midface_score = float(np.clip(midface_score, 1.0, 10.0))

        ideal_jaw = 0.92 if gender == "male" else 0.84
        jaw_score = 9.5 - abs(jaw_ratio - ideal_jaw) * 16
        jaw_score = float(np.clip(jaw_score, 2.0, 9.5))

        sym_score = 9.5 - sym_err * 0.4
        sym_score = float(np.clip(sym_score, 3.0, 9.8))

        eyes_score = float(np.clip(canthal_score * 0.9 + 0.6, 1.0, 10.0))
        cheek_score = float(np.clip(6.2 + (jaw_ratio - 0.8) * 6, 3.0, 9.2))
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
            "mp_available": False,
            "overlays": {},
        }
