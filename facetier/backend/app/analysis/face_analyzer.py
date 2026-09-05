"""FaceTier MediaPipe analyzer — landmarks locked to real face geometry."""
from typing import Dict, Any, Optional, List, Tuple
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
        self.face_mesh_soft = None
        if MP_AVAILABLE:
            # Primary: accurate mesh
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            # Soft fallback for difficult lighting / angles
            self.face_mesh_soft = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def analyze(self, front_path: str, profile_path: Optional[str] = None,
                gender: str = "male", age: Optional[int] = None) -> Dict[str, Any]:
        if not MP_AVAILABLE:
            r = self._fallback_result(gender)
            r["warning"] = f"MediaPipe недоступен: {MP_IMPORT_ERROR}"
            r["mp_available"] = False
            return r

        front_img = cv2.imread(front_path)
        if front_img is None:
            raise ValueError("Не удалось прочитать фото анфаса")

        h, w = front_img.shape[:2]
        rgb = self._preprocess(front_img)

        landmarks = self._run_mesh(rgb)
        if landmarks is None:
            raise ValueError(
                "Лицо на анфасе не найдено. "
                "Сфотографируй лицо прямо, крупно, при хорошем освещении, без сильного наклона."
            )

        pts_n = {i: (lm.x, lm.y) for i, lm in enumerate(landmarks.landmark)}
        pts_px = {i: (lm.x * w, lm.y * h) for i, lm in enumerate(landmarks.landmark)}

        metrics = self._compute_metrics(pts_px, w, h)
        scores = self._metrics_to_scores(metrics, gender)
        overlays = self._build_overlays(pts_n)

        profile_ok = False
        if profile_path:
            profile_img = cv2.imread(profile_path)
            if profile_img is not None:
                prof_ov, profile_ok = self._analyze_profile(profile_img)
                overlays["profile"] = prof_ov

        overall = round(float(np.mean(list(scores.values()))), 1)
        potential = round(min(9.5, overall + 1.3), 1)

        return {
            "metrics": metrics,
            "scores": scores,
            "overlays": overlays,
            "overall_score": overall,
            "potential_score": potential,
            "mp_available": True,
            "profile_mesh": profile_ok,
        }

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Light CLAHE + mild denoise so MediaPipe locks better on real face."""
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            # mild bilateral to keep edges but reduce noise
            enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)
            return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        except Exception:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _run_mesh(self, rgb: np.ndarray):
        for mesh in (self.face_mesh, self.face_mesh_soft):
            if mesh is None:
                continue
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                return res.multi_face_landmarks[0]
        return None

    def _analyze_profile(self, profile_img):
        rgb = self._preprocess(profile_img)
        landmarks = self._run_mesh(rgb)
        if landmarks is not None:
            pts = {i: (p.x, p.y) for i, p in enumerate(landmarks.landmark)}
            return self._build_profile_overlays(pts), True

        # last resort: face detection bbox → relative bands (still face-relative)
        bbox = self._detect_face_bbox(rgb)
        if bbox:
            return self._build_profile_from_bbox(bbox), False
        return self._build_profile_default_bands(), False

    def _detect_face_bbox(self, rgb):
        try:
            det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.25
            )
            out = det.process(rgb)
            det.close()
            if not out.detections:
                return None
            d = out.detections[0].location_data.relative_bounding_box
            return [
                max(0.0, d.xmin),
                max(0.0, d.ymin),
                min(1.0, d.xmin + d.width),
                min(1.0, d.ymin + d.height),
            ]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Profile overlays (mesh-locked)
    # ------------------------------------------------------------------
    def _build_profile_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        fh = max(y2 - y1, 0.05)
        fw = max(x2 - x1, 0.05)

        def band(y_rel0, y_rel1, x_pad=0.04):
            ya = y1 + fh * y_rel0
            yb = y1 + fh * y_rel1
            xa = max(0.0, x1 - fw * x_pad)
            xb = min(1.0, x2 + fw * x_pad)
            return [
                [round(xa, 4), round(ya, 4)],
                [round(xb, 4), round(ya, 4)],
                [round(xb, 4), round(yb, 4)],
                [round(xa, 4), round(yb, 4)],
            ]

        return {
            "midface": {"kind": "polygon", "points": band(0.16, 0.48), "color": "#7c5cff", "source": "bbox"},
            "jaw": {"kind": "polygon", "points": band(0.40, 0.72), "color": "#ffb020", "source": "bbox"},
            "chin": {"kind": "polygon", "points": band(0.64, 0.96, 0.10), "color": "#00d4aa", "source": "bbox"},
            "overall": {
                "kind": "polygon",
                "points": [
                    [round(x1, 4), round(y1, 4)],
                    [round(x2, 4), round(y1, 4)],
                    [round(x2, 4), round(y2, 4)],
                    [round(x1, 4), round(y2, 4)],
                ],
                "color": "#7c5cff",
                "source": "bbox",
            },
        }

    def _build_profile_default_bands(self):
        def rect(y0, y1):
            return [[0.10, y0], [0.90, y0], [0.90, y1], [0.10, y1]]

        return {
            "midface": {"kind": "polygon", "points": rect(0.20, 0.46), "color": "#7c5cff", "source": "default"},
            "jaw": {"kind": "polygon", "points": rect(0.42, 0.70), "color": "#ffb020", "source": "default"},
            "chin": {"kind": "polygon", "points": rect(0.64, 0.88), "color": "#00d4aa", "source": "default"},
        }

    def _build_profile_overlays(self, pts):
        """All points come from MediaPipe → locked to the actual face."""
        # Full lower face contour (jaw + chin)
        jaw_line = self._pts(pts, [
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397
        ])
        # Tight chin region around menton
        chin_pts = self._pts(pts, [152, 175, 199, 208, 32, 140, 176, 148, 377, 400, 378])
        # Midface vertical span from brow → mouth
        brow_y = self._avg_y(pts, [70, 105, 300, 334])
        eye_y = self._avg_y(pts, [33, 133, 263, 362, 159, 386])
        nose_y = self._avg_y(pts, [1, 2, 4, 5, 6])
        mouth_y = self._avg_y(pts, [13, 14, 0, 17])
        chin_y = self._avg_y(pts, [152, 175, 199])

        # Horizontal bounds from cheek / jaw points
        xs = [p[0] for p in self._pts(pts, [234, 454, 93, 323, 132, 361, 58, 288, 172, 397])]
        if not xs:
            xs = [0.25, 0.75]
        x_min = max(0.0, min(xs) - 0.03)
        x_max = min(1.0, max(xs) + 0.03)

        def hband(y0, y1):
            y0 = float(np.clip(y0, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            if y1 <= y0:
                y1 = min(1.0, y0 + 0.08)
            return [
                [round(x_min, 4), round(y0, 4)],
                [round(x_max, 4), round(y0, 4)],
                [round(x_max, 4), round(y1, 4)],
                [round(x_min, 4), round(y1, 4)],
            ]

        mid_top = brow_y if brow_y is not None else (eye_y - 0.05 if eye_y else 0.20)
        mid_bot = mouth_y if mouth_y is not None else (nose_y + 0.07 if nose_y else 0.48)
        if mid_bot <= mid_top:
            mid_bot = mid_top + 0.16

        jaw_top = mouth_y if mouth_y is not None else 0.48
        jaw_bot = chin_y if chin_y is not None else 0.80
        if jaw_bot <= jaw_top:
            jaw_bot = jaw_top + 0.18

        chin_top = max(jaw_top, (chin_y - 0.07) if chin_y else (jaw_top + jaw_bot) / 2)
        chin_bot = min(0.99, (chin_y + 0.07) if chin_y else jaw_bot + 0.05)

        jaw_ov = (
            {"kind": "polyline", "points": jaw_line, "color": "#ffb020", "source": "mesh"}
            if len(jaw_line) >= 4
            else {"kind": "polygon", "points": hband(jaw_top, jaw_bot), "color": "#ffb020", "source": "mesh"}
        )
        chin_ov = (
            {
                "kind": "points+lines",
                "points": chin_pts,
                "lines": [chin_pts] if len(chin_pts) >= 2 else [],
                "color": "#00d4aa",
                "source": "mesh",
            }
            if chin_pts
            else {"kind": "polygon", "points": hband(chin_top, chin_bot), "color": "#00d4aa", "source": "mesh"}
        )

        face_outline = self._pts(pts, [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        ])

        return {
            "midface": {"kind": "polygon", "points": hband(mid_top, mid_bot), "color": "#7c5cff", "source": "mesh"},
            "jaw": jaw_ov,
            "chin": chin_ov,
            "overall": {
                "kind": "polyline",
                "points": face_outline,
                "color": "#7c5cff",
                "source": "mesh",
            },
        }

    # ------------------------------------------------------------------
    # Front overlays — pure MediaPipe landmarks (locked to face)
    # ------------------------------------------------------------------
    def _build_overlays(self, pts):
        # Eyes + canthal
        eyes_pts = self._pts(pts, [33, 133, 263, 362, 468, 473, 159, 386, 70, 300])
        # Midface: glabella → subnasale → mouth, cheek width
        midface_pts = self._pts(pts, [10, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 94, 19, 13, 14])
        # Jawline full contour
        jaw_pts = self._pts(pts, [
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323
        ])
        # Cheekbones (zygomatic)
        cheek_pts = self._pts(pts, [234, 93, 132, 58, 172, 454, 323, 361, 288, 397, 116, 345, 123, 352])
        # Chin — tight around menton + lower contour
        chin_pts = self._pts(pts, [152, 175, 199, 208, 32, 140, 176, 148, 377, 400, 378, 379])
        # Full face oval
        face_outline = self._pts(pts, [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        ])
        # Symmetry midline
        center_line = self._pts(pts, [10, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 13, 14, 17, 18, 200, 152])

        return {
            "front": {
                "overall": {
                    "kind": "polyline",
                    "points": face_outline,
                    "color": "#7c5cff",
                },
                "eyes": {
                    "kind": "points+lines",
                    "points": eyes_pts,
                    "lines": [
                        self._pts(pts, [33, 133]),
                        self._pts(pts, [263, 362]),
                        self._pts(pts, [33, 263]),  # intercanthal reference
                    ],
                    "color": "#00d4aa",
                },
                "midface": {
                    "kind": "points+lines",
                    "points": midface_pts,
                    "lines": [
                        self._pts(pts, [10, 152]),          # vertical midline of midface
                        self._pts(pts, [234, 454]),         # cheek width
                        self._pts(pts, [98, 327]),          # nose width
                    ],
                    "color": "#7c5cff",
                },
                "jaw": {
                    "kind": "polyline",
                    "points": jaw_pts,
                    "color": "#ffb020",
                },
                "cheekbones": {
                    "kind": "points+lines",
                    "points": cheek_pts,
                    "lines": [self._pts(pts, [234, 454])],
                    "color": "#ff5c7a",
                },
                "chin": {
                    "kind": "points+lines",
                    "points": chin_pts,
                    "lines": [chin_pts] if len(chin_pts) >= 2 else [],
                    "color": "#00d4aa",
                },
                "symmetry": {
                    "kind": "points+lines",
                    "points": center_line + eyes_pts[:4],
                    "lines": [
                        center_line,
                        self._pts(pts, [33, 263]),
                        self._pts(pts, [234, 454]),
                    ],
                    "color": "#7c5cff",
                },
                "harmony": {
                    "kind": "polyline",
                    "points": face_outline,
                    "color": "#00d4aa",
                },
            },
            # profile will be filled later if photo exists
            "profile": {},
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _avg_y(self, pts, ids):
        ys = [pts[i][1] for i in ids if i in pts]
        return float(np.mean(ys)) if ys else None

    def _dist(self, p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _n(self, pts, idx):
        p = pts.get(idx)
        return [round(float(p[0]), 4), round(float(p[1]), 4)] if p else None

    def _pts(self, pts, ids):
        out = []
        for i in ids:
            p = self._n(pts, i)
            if p:
                out.append(p)
        return out

    def _compute_metrics(self, pts_px, w, h):
        def safe(i):
            return pts_px.get(i)

        canthal = 0.0
        angles = []
        left_o, left_i = safe(33), safe(133)
        right_o, right_i = safe(263), safe(362)
        if left_o and left_i and abs(left_i[0] - left_o[0]) > 1e-6:
            angles.append(
                float(np.degrees(np.arctan2(-(left_o[1] - left_i[1]), abs(left_i[0] - left_o[0]))))
            )
        if right_o and right_i and abs(right_o[0] - right_i[0]) > 1e-6:
            angles.append(
                float(np.degrees(np.arctan2(-(right_o[1] - right_i[1]), abs(right_o[0] - right_i[0]))))
            )
        if angles:
            canthal = float(np.mean(angles))

        top, subnasale, menton = safe(10), safe(2), safe(152)
        midface_ratio = 1.0
        if top and subnasale and menton:
            upper = self._dist(top, subnasale)
            lower = self._dist(subnasale, menton)
            if lower > 1e-6:
                midface_ratio = float(upper / lower)

        lj, rj, lc, rc = safe(172), safe(397), safe(234), safe(454)
        jaw_to_cheek = 0.85
        if lj and rj and lc and rc:
            jw = self._dist(lj, rj)
            cw = self._dist(lc, rc)
            if cw > 1e-6:
                jaw_to_cheek = float(jw / cw)

        midline_x = None
        if safe(10) and safe(152):
            midline_x = (safe(10)[0] + safe(152)[0]) / 2.0
        errs = []
        if midline_x is not None:
            for li, ri in [(33, 263), (133, 362), (234, 454), (172, 397), (61, 291)]:
                lp, rp = safe(li), safe(ri)
                if lp and rp:
                    errs.append(
                        abs((midline_x - lp[0]) - (rp[0] - midline_x)) / max(w, 1) * 100.0
                    )

        return {
            "canthal_tilt": round(canthal, 2),
            "midface_ratio": round(midface_ratio, 3),
            "jaw_to_cheek": round(jaw_to_cheek, 3),
            "symmetry_error": round(float(np.mean(errs)) if errs else 5.0, 2),
        }

    def _metrics_to_scores(self, metrics, gender):
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
        jaw_score = float(np.clip(9.5 - abs(jaw_ratio - ideal_jaw) * 16, 2.0, 9.5))
        sym_score = float(np.clip(9.5 - sym_err * 0.4, 3.0, 9.8))
        eyes_score = float(np.clip(canthal_score * 0.9 + 0.6, 1.0, 10.0))
        cheek_score = float(np.clip(6.2 + (jaw_ratio - 0.8) * 6, 3.0, 9.2))
        chin_score = float(np.clip(jaw_score * 0.95, 2.5, 9.2))
        harmony = float(np.mean([canthal_score, midface_score, jaw_score, sym_score]))

        return {
            k: round(v, 1)
            for k, v in {
                "canthal_tilt": canthal_score,
                "eyes": eyes_score,
                "midface": midface_score,
                "jaw": jaw_score,
                "cheekbones": cheek_score,
                "chin": chin_score,
                "symmetry": sym_score,
                "harmony": harmony,
            }.items()
        }

    def _fallback_result(self, gender):
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
            "profile_mesh": False,
            "overlays": {},
        }
