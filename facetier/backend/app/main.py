from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import tempfile

app = FastAPI(title="FaceTier API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "facetier_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"status": "ok", "service": "FaceTier API", "version": "0.3.1"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug")
async def debug():
    info = {"version": "0.3.1"}
    try:
        from app.analysis import face_analyzer as fa
        info["mp_available"] = fa.MP_AVAILABLE
        info["mp_import_error"] = fa.MP_IMPORT_ERROR
        info["cv2"] = True
    except Exception as e:
        info["mp_available"] = False
        info["mp_import_error"] = str(e)
        info["cv2"] = False
    try:
        import mediapipe
        info["mediapipe_version"] = getattr(mediapipe, "__version__", "unknown")
    except Exception as e:
        info["mediapipe_version"] = None
        info["mediapipe_error"] = str(e)
    try:
        import cv2
        info["cv2_version"] = cv2.__version__
    except Exception as e:
        info["cv2_error"] = str(e)
    return info


@app.post("/api/analyze")
async def analyze_face(
    front: UploadFile = File(...),
    profile: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(None),
):
    if gender not in ("male", "female"):
        raise HTTPException(400, "gender must be 'male' or 'female'")

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    front_path = session_dir / "front.jpg"
    profile_path = session_dir / "profile.jpg"

    try:
        with open(front_path, "wb") as f:
            shutil.copyfileobj(front.file, f)
        with open(profile_path, "wb") as f:
            shutil.copyfileobj(profile.file, f)

        try:
            from app.analysis.face_analyzer import FaceAnalyzer
            analyzer = FaceAnalyzer()
            analysis = analyzer.analyze(
                str(front_path),
                str(profile_path),
                gender=gender,
                age=age,
            )
            scores = analysis.get("scores", {})
            overall = analysis.get("overall_score", 6.0)
            potential = analysis.get("potential_score", 7.5)
            metrics = analysis.get("metrics", {})
            mp_ok = analysis.get("mp_available", False)
            warning = analysis.get("warning")

            canthal_score = scores.get("canthal_tilt", 6.5)
            canthal_value = float(metrics.get("canthal_tilt", 0))
            left_c = metrics.get("left_canthal", 0)
            right_c = metrics.get("right_canthal", 0)

            msg = "Анализ выполнен через MediaPipe landmarks" if mp_ok else f"Демо-режим: {warning or 'MediaPipe недоступен'}"

            result = {
                "session_id": session_id,
                "overall_score": overall,
                "potential_score": potential,
                "gender": gender,
                "age": age,
                "mp_available": mp_ok,
                "metrics": metrics,
                "demo_parameter": {
                    "name": "Canthal Tilt",
                    "name_ru": "Кантальный наклон",
                    "score": canthal_score,
                    "potential": min(9.5, canthal_score + 1.2),
                    "value": f"{canthal_value:+.1f}°",
                    "left": f"{float(left_c):+.1f}°",
                    "right": f"{float(right_c):+.1f}°",
                    "description": _canthal_description(canthal_value),
                    "advice": _canthal_advice(canthal_value),
                },
                "zones": [
                    {"id": "eyes", "name": "Глаза", "score": scores.get("eyes", 6.5), "potential": min(9.2, scores.get("eyes", 6.5) + 1.1)},
                    {"id": "midface", "name": "Midface", "score": scores.get("midface", 6.0), "potential": min(8.8, scores.get("midface", 6.0) + 1.3)},
                    {"id": "jaw", "name": "Челюсть", "score": scores.get("jaw", 5.8), "potential": min(9.0, scores.get("jaw", 5.8) + 1.6)},
                    {"id": "cheekbones", "name": "Скулы", "score": scores.get("cheekbones", 6.3), "potential": min(8.7, scores.get("cheekbones", 6.3) + 1.2)},
                    {"id": "chin", "name": "Подбородок", "score": scores.get("chin", 6.0), "potential": min(8.5, scores.get("chin", 6.0) + 1.1)},
                    {"id": "symmetry", "name": "Симметрия", "score": scores.get("symmetry", 6.8), "potential": min(8.5, scores.get("symmetry", 6.8) + 0.8)},
                    {"id": "harmony", "name": "Гармония", "score": scores.get("harmony", 6.2), "potential": min(8.6, scores.get("harmony", 6.2) + 1.2)},
                ],
                "priorities": _build_priorities(scores),
                "message": msg,
            }
            if warning:
                result["warning"] = warning
        except Exception as e:
            result = _demo_result(session_id, gender, age, error=str(e))

        return JSONResponse(result)

    finally:
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


def _canthal_description(value: float) -> str:
    if value >= 5:
        return "Сильный положительный кантальный наклон. Глаза выглядят хищными и открытыми (hunter eyes vibe)."
    if value >= 2:
        return "Положительный кантальный наклон. Хороший показатель, глаза выглядят живыми."
    if value >= -1:
        return "Нейтральный / слабо положительный наклон. Средний показатель."
    return "Отрицательный кантальный наклон. Глаза визуально выглядят более усталыми."


def _canthal_advice(value: float) -> list:
    if value >= 4:
        return [
            "Отличный показатель. Поддерживай его формой бровей и хорошим сном.",
            "Избегай сильной отёчности — она может визуально снижать эффект.",
        ]
    if value >= 1:
        return [
            "Хороший базовый уровень. Можно чуть усилить впечатление правильной формой бровей.",
            "Следи за сном и отёками под глазами.",
        ]
    return [
        "Есть потенциал. Форма бровей и работа с нижним веком могут визуально улучшить восприятие.",
        "В долгосрочной перспективе некоторые рассматривают кантопластику, но это уже хирургия.",
    ]


def _build_priorities(scores: dict) -> list:
    items = [
        ("Челюсть", scores.get("jaw", 5)),
        ("Midface", scores.get("midface", 5)),
        ("Гармония", scores.get("harmony", 5)),
        ("Скулы", scores.get("cheekbones", 5)),
        ("Глаза", scores.get("eyes", 5)),
    ]
    items.sort(key=lambda x: x[1])
    return [f"{name} — наибольший потенциал улучшения" for name, _ in items[:3]]


def _demo_result(session_id, gender, age, error=None):
    return {
        "session_id": session_id,
        "overall_score": 6.3,
        "potential_score": 7.7,
        "gender": gender,
        "age": age,
        "mp_available": False,
        "demo_parameter": {
            "name": "Canthal Tilt",
            "name_ru": "Кантальный наклон",
            "score": 6.5,
            "potential": 8.0,
            "value": "+2.1°",
            "description": "Нейтрально-положительный наклон (демо-режим).",
            "advice": ["Это демо-результат, потому что анализ не смог выполниться полностью."],
        },
        "zones": [
            {"id": "eyes", "name": "Глаза", "score": 6.5, "potential": 8.0},
            {"id": "midface", "name": "Midface", "score": 6.0, "potential": 7.5},
            {"id": "jaw", "name": "Челюсть", "score": 5.8, "potential": 7.9},
            {"id": "cheekbones", "name": "Скулы", "score": 6.3, "potential": 7.8},
            {"id": "chin", "name": "Подбородок", "score": 6.0, "potential": 7.4},
            {"id": "symmetry", "name": "Симметрия", "score": 6.8, "potential": 7.6},
            {"id": "harmony", "name": "Гармония", "score": 6.2, "potential": 7.7},
        ],
        "priorities": [
            "Челюсть — наибольший потенциал улучшения",
            "Midface — стоит уделить внимание",
            "Гармония лица",
        ],
        "message": f"Демо-режим. Ошибка анализа: {error}" if error else "Демо-режим",
    }
