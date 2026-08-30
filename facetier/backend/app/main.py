from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import tempfile

app = FastAPI(title="FaceTier API", version="0.3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "facetier_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ZONE_DAILY = {
    "eyes": {
        "name": "Глаза",
        "photo": "front",
        "daily": ["Сон 7–9 часов", "Холод на область глаз утром 1–2 мин", "Не тереть глаза, не спать лицом в подушку"],
        "weekly": ["Проверь форму бровей", "Фото анфас при одном свете"],
    },
    "midface": {
        "name": "Midface",
        "photo": "profile",
        "daily": ["Mewing: язык на нёбе, рот закрыт", "Осанка: уши над плечами", "Дыши носом"],
        "weekly": ["Фото профиля раз в неделю", "Контроль веса если жир на лице"],
    },
    "jaw": {
        "name": "Челюсть",
        "photo": "profile",
        "daily": ["Жвачка 10–15 мин без сахара", "Mewing + сомкнутые губы", "Меньше соли и алкоголя вечером"],
        "weekly": ["Силовые + дефицит при лишнем весе", "Не спи лицом в подушку"],
    },
    "cheekbones": {
        "name": "Скулы",
        "photo": "front",
        "daily": ["Сон и вода", "Не раздувай лицо перееданием на ночь"],
        "weekly": ["Сушка лица через общий процент жира"],
    },
    "chin": {
        "name": "Подбородок",
        "photo": "profile",
        "daily": ["Осанка", "Язык на нёбе, рот закрыт"],
        "weekly": ["Фото строгого профиля"],
    },
    "symmetry": {
        "name": "Симметрия",
        "photo": "front",
        "daily": ["Жуй на обе стороны", "Не всегда спи на одном боку"],
        "weekly": ["Сравни зеркальные фото"],
    },
    "harmony": {
        "name": "Гармония",
        "photo": "front",
        "daily": ["Сон, вода, mewing, осанка — база", "Один главный фокус из приоритетов"],
        "weekly": ["3 фото при одном свете: анфас, 45°, профиль"],
    },
}


def _level(score: float) -> str:
    if score >= 8.0:
        return "strong"
    if score >= 6.5:
        return "ok"
    if score >= 5.0:
        return "weak"
    return "critical"


def _canthal_description(value: float) -> str:
    if value >= 5:
        return "Сильный положительный кантальный наклон. Hunter eyes vibe."
    if value >= 2:
        return "Положительный кантальный наклон. Хороший показатель."
    if value >= -1:
        return "Нейтральный / слабо положительный наклон. Средний показатель."
    return "Отрицательный кантальный наклон. Глаза выглядят более усталыми."


def _canthal_advice(value: float) -> list:
    if value >= 4:
        return ["Держи сон и форму бровей.", "Избегай отёков."]
    if value >= 1:
        return ["Форма бровей может усилить взгляд.", "Следи за сном."]
    return ["Брови и нижнее веко влияют сильнее, чем кажется.", "Сначала дисциплина 2–3 месяца."]


@app.get("/")
async def root():
    return {"status": "ok", "service": "FaceTier API", "version": "0.3.2"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug")
async def debug():
    info = {"version": "0.3.2"}
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
            analysis = analyzer.analyze(str(front_path), str(profile_path), gender=gender, age=age)
            scores = analysis.get("scores", {})
            overall = analysis.get("overall_score", 6.0)
            potential = analysis.get("potential_score", 7.5)
            metrics = analysis.get("metrics", {})
            overlays = analysis.get("overlays") or {}
            mp_ok = analysis.get("mp_available", False)
            warning = analysis.get("warning")

            canthal_score = float(scores.get("canthal_tilt", 6.5))
            canthal_value = float(metrics.get("canthal_tilt", 0))
            left_c = float(metrics.get("left_canthal", 0))
            right_c = float(metrics.get("right_canthal", 0))

            zones = []
            for zid, meta in ZONE_DAILY.items():
                sc = float(scores.get(zid, 6.0))
                pot = min(9.5, round(sc + (1.5 if sc < 6 else 1.1), 1))
                lvl = _level(sc)
                desc = _canthal_description(canthal_value) if zid == "eyes" else f"Оценка {sc:.1f}/10. Потенциал {pot:.1f}."
                if zid == "eyes":
                    desc = f"Canthal {canthal_value:+.1f}° (L {left_c:+.1f} / R {right_c:+.1f}). " + desc
                zones.append({
                    "id": zid,
                    "name": meta["name"],
                    "score": sc,
                    "potential": pot,
                    "level": lvl,
                    "description": desc,
                    "daily": list(meta["daily"]),
                    "weekly": list(meta["weekly"]),
                    "photo": meta["photo"],
                    "overlay_key": zid,
                })

            weak = sorted(zones, key=lambda z: z["score"])[:3]
            daily_plan = {
                "base": [
                    "Сон 7–9 часов",
                    "Рот закрыт, дыхание носом, язык к нёбу",
                    "Осанка: уши над плечами",
                    "Вода днём, меньше соли вечером",
                ],
                "focus_today": [f"{z['name']}: {z['daily'][0]}" for z in weak],
                "note": "Сначала база каждый день. Потом 1–2 фокуса из слабых зон.",
            }
            priorities = [f"{z['name']} ({z['score']:.1f}) — приоритет" for z in weak]
            gap = round(potential - overall, 1)
            summary = f"Оценка {overall:.1f}/10, потенциал ~{potential:.1f} (запас {gap:.1f}). Фокус: {weak[0]['name'] if weak else 'база'}."
            msg = "Полный разбор через MediaPipe" if mp_ok else f"Демо-режим: {warning or 'MediaPipe недоступен'}"

            result = {
                "session_id": session_id,
                "overall_score": overall,
                "potential_score": potential,
                "gender": gender,
                "age": age,
                "mp_available": mp_ok,
                "metrics": metrics,
                "overlays": overlays,
                "summary": summary,
                "daily_plan": daily_plan,
                "demo_parameter": {
                    "name": "Canthal Tilt",
                    "name_ru": "Кантальный наклон",
                    "score": canthal_score,
                    "potential": min(9.5, canthal_score + 1.2),
                    "value": f"{canthal_value:+.1f}°",
                    "left": f"{left_c:+.1f}°",
                    "right": f"{right_c:+.1f}°",
                    "description": _canthal_description(canthal_value),
                    "advice": _canthal_advice(canthal_value),
                },
                "zones": zones,
                "priorities": priorities,
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


def _demo_result(session_id, gender, age, error=None):
    scores = {"eyes": 6.5, "midface": 6.0, "jaw": 5.8, "cheekbones": 6.3, "chin": 6.0, "symmetry": 6.8, "harmony": 6.2}
    zones = []
    for zid, meta in ZONE_DAILY.items():
        sc = scores[zid]
        zones.append({
            "id": zid, "name": meta["name"], "score": sc, "potential": sc + 1.2,
            "level": _level(sc), "description": "Демо-режим",
            "daily": meta["daily"], "weekly": meta["weekly"],
            "photo": meta["photo"], "overlay_key": zid,
        })
    return {
        "session_id": session_id,
        "overall_score": 6.3,
        "potential_score": 7.7,
        "gender": gender,
        "age": age,
        "mp_available": False,
        "overlays": {},
        "summary": "Демо-режим. Загрузи чёткий анфас для реального разбора.",
        "daily_plan": {
            "base": ["Сон 7–9 часов", "Mewing", "Осанка", "Вода"],
            "focus_today": ["Держи базу"],
            "note": "",
        },
        "demo_parameter": {
            "name": "Canthal Tilt", "name_ru": "Кантальный наклон",
            "score": 6.5, "potential": 8.0, "value": "+2.1°",
            "description": "Демо.", "advice": ["Демо."],
        },
        "zones": zones,
        "priorities": ["Челюсть — приоритет", "Midface — приоритет"],
        "message": f"Демо-режим. Ошибка: {error}" if error else "Демо-режим",
    }
