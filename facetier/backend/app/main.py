from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import tempfile

app = FastAPI(title="FaceTier API", version="0.7.0")

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
    return {"status": "ok", "service": "FaceTier API", "version": "0.7.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug")
async def debug():
    info = {"version": "0.7.0"}
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
            scores = analysis.get("scores") or {}
            metrics = analysis.get("metrics") or {}
            overall = analysis.get("overall_score", 6.0)
            potential = analysis.get("potential_score", overall + 1.0)

            zones = _build_full_zones(scores, metrics, gender)
            daily = _build_daily_plan(zones, gender)
            priorities = _build_priorities(scores)
            summary = _build_summary(overall, potential, priorities, gender)

            for z in zones:
                zid = z["id"]
                z["photo"] = "profile" if zid in ("jaw", "chin", "midface") else "front"
                z["overlay_key"] = "eyes" if zid == "eyes" else zid

            return {
                "session_id": session_id,
                "overall_score": overall,
                "potential_score": potential,
                "gender": gender,
                "age": age,
                "mp_available": analysis.get("mp_available", False),
                "overlays": analysis.get("overlays") or {},
                "profile_mesh": analysis.get("profile_mesh", False),
                "zones": zones,
                "daily_plan": daily,
                "priorities": priorities,
                "summary": summary,
                "metrics": metrics,
                "message": "OK",
            }
        except Exception as e:
            return _demo_result(session_id, gender, age, error=str(e))
    finally:
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass


def _build_full_zones(scores: dict, metrics: dict, gender: str) -> list:
    def zone(zid, name, score_key, potential_extra, desc_fn, daily_fn, weekly_fn):
        sc = float(scores.get(score_key, 6.0))
        return {
            "id": zid,
            "name": name,
            "score": sc,
            "potential": round(min(9.8, sc + potential_extra), 1),
            "description": desc_fn(sc, metrics),
            "daily": daily_fn(sc),
            "weekly": weekly_fn(sc),
        }

    eyes_desc = lambda s, m: (
        f"Canthal tilt ~{m.get('canthal_tilt', 0):.1f}°. "
        + ("Хороший наклон, держи форму." if s >= 7.5 else "Есть куда расти — угол и раскрытие." if s >= 5.5 else "Слабый canthal / форма глаз тянет вниз.")
    )
    eyes_daily = lambda s: (
        ["Сон 7–8ч, холодные компрессы утром", "Лёгкий массаж периорбитальной зоны", "Не тереть глаза, снижай отёки"]
        if s < 7
        else ["Поддерживай сон и гигиену кожи вокруг глаз", "Защита от UV"]
    )
    eyes_weekly = lambda s: ["Фотоконтроль canthal раз в 2 недели", "Если сильный hooding — консультация"]

    mid_desc = lambda s, m: (
        f"Midface ratio ~{m.get('midface_ratio', 1):.2f}. "
        + ("Баланс ок." if 0.95 <= m.get('midface_ratio', 1) <= 1.25 else "Средняя треть смещена — работай над объёмом и привычками.")
    )
    mid_daily = lambda s: [
        "Mewing: язык к нёбу весь день",
        "Дыши носом, язык к нёбу (mewing) — базовый daily",
        "Жуй твёрдую пищу, меньше жидкости перед сном",
    ]
    mid_weekly = lambda s: ["Проверка осанки и дыхания", "Фото midface анфас + 3/4"]

    jaw_desc = lambda s, m: (
        f"Jaw/cheek ~{m.get('jaw_to_cheek', 0.85):.2f}. "
        + ("Челюсть заметная." if s >= 7 else "Челюсть слабая — приоритет №1." if s < 5.5 else "Есть потенциал для угла и ширины.")
    )
    jaw_daily = lambda s: [
        "Жевательная нагрузка (жвачка / твёрдая еда) 20–40 мин",
        "Neck curls / chin tucks 2×15",
        "Mewing + правильная осанка",
    ]
    jaw_weekly = lambda s: ["Фото профиля и анфаса челюсти", "Прогресс по углу gonial"]

    cheek_desc = lambda s, m: "Скулы читаются по ширине и проекции." if s >= 6.5 else "Скулы слабые — визуал + объём."
    cheek_daily = lambda s: ["Массаж скул вверх", "Не спать лицом в подушку"]
    cheek_weekly = lambda s: ["Контроль жира в лице"]

    chin_desc = lambda s, m: "Подбородок в балансе." if s >= 7 else "Подбородок слабый / рецессивный — приоритет."
    chin_daily = lambda s: [
        "Chin tucks 3×15",
        "Не выдвигать голову вперёд",
        "При сильном рецессе подбородка долгосрочно рассматривают импланты/гениопластику — это уже не daily",
    ]
    chin_weekly = lambda s: ["Фото профиля подбородка"]

    sym_desc = lambda s, m: f"Асимметрия ~{m.get('symmetry_error', 5):.1f}%. " + ("В пределах нормы." if s >= 7 else "Заметный перекос — привычки и осанка.")
    sym_daily = lambda s: [
        "Жуй обеими сторонами",
        "Спи ровно, не на одном боку постоянно",
        "Сильную скелетную асимметрию daily не исправит — только визуал и привычки",
    ]
    sym_weekly = lambda s: ["Сравни лево/право на фото"]

    harm_desc = lambda s, m: "Гармония зон." if s >= 7 else "Зоны спорят друг с другом — выравнивай слабые."
    harm_daily = lambda s: ["Общая дисциплина: сон, осанка, mewing", "Не читерь одну зону в ущерб другим"]
    harm_weekly = lambda s: ["Полный разбор раз в 2–4 недели"]

    return [
        zone("eyes", "Глаза / Canthal", "eyes", 1.1, eyes_desc, eyes_daily, eyes_weekly),
        zone("midface", "Midface", "midface", 1.3, mid_desc, mid_daily, mid_weekly),
        zone("jaw", "Челюсть", "jaw", 1.6, jaw_desc, jaw_daily, jaw_weekly),
        zone("cheekbones", "Скулы", "cheekbones", 1.2, cheek_desc, cheek_daily, cheek_weekly),
        zone("chin", "Подбородок", "chin", 1.1, chin_desc, chin_daily, chin_weekly),
        zone("symmetry", "Симметрия", "symmetry", 0.8, sym_desc, sym_daily, sym_weekly),
        zone("harmony", "Гармония", "harmony", 1.2, harm_desc, harm_daily, harm_weekly),
    ]


def _build_daily_plan(zones: list, gender: str) -> dict:
    """Собрать общий daily plan из самых слабых зон + база."""
    base = [
        "Сон 7–8 часов",
        "Mewing + носовое дыхание весь день",
        "Вода, меньше сахара и алкоголя",
        "Осанка: уши над плечами, язык к нёбу",
    ]
    weak = sorted(zones, key=lambda z: z["score"])[:3]
    focus = []
    for z in weak:
        focus.append(f"{z['name']}: {z['daily'][0]}")
        if len(z["daily"]) > 1:
            focus.append(f"{z['name']}: {z['daily'][1]}")

    return {
        "base": base,
        "focus_today": focus[:6],
        "note": "Сначала база каждый день. Потом 1–2 фокуса из слабых зон — не всё сразу.",
    }


def _build_priorities(scores: dict) -> list:
    items = [
        ("Челюсть", scores.get("jaw", 5)),
        ("Midface", scores.get("midface", 5)),
        ("Глаза / Canthal", scores.get("eyes", 5)),
        ("Подбородок", scores.get("chin", 5)),
        ("Скулы", scores.get("cheekbones", 5)),
        ("Симметрия", scores.get("symmetry", 5)),
        ("Гармония", scores.get("harmony", 5)),
    ]
    items.sort(key=lambda x: x[1])
    out = []
    for name, sc in items[:4]:
        if sc < 5.5:
            out.append(f"{name} ({sc:.1f}) — критичный приоритет")
        elif sc < 7:
            out.append(f"{name} ({sc:.1f}) — главный рычаг роста")
        else:
            out.append(f"{name} ({sc:.1f}) — поддерживать")
    return out


def _build_summary(overall: float, potential: float, priorities: list, gender: str) -> str:
    gap = potential - overall
    tone = (
        "Сильная база. Дальше — точечная доводка и дисциплина."
        if overall >= 7.5
        else "Средний уровень. Рост реально упирается в 2–3 слабые зоны и daily-привычки."
        if overall >= 5.5
        else "Сейчас лицо тянут вниз слабые зоны. Не распыляйся: база + топ-приоритеты."
    )
    top = priorities[0] if priorities else "дисциплина"
    return (
        f"Оценка {overall:.1f}/10, потенциал около {potential:.1f} (запас ~{gap:.1f}). "
        f"{tone} Ближайший фокус: {top}."
    )


def _demo_result(session_id, gender, age, error=None):
    scores = {
        "canthal_tilt": 6.5,
        "eyes": 6.5,
        "midface": 6.0,
        "jaw": 5.8,
        "cheekbones": 6.3,
        "chin": 6.0,
        "symmetry": 6.8,
        "harmony": 6.2,
    }
    zones = _build_full_zones(scores, {"canthal_tilt": 0, "midface_ratio": 1.0, "jaw_to_cheek": 0.85, "symmetry_error": 5}, gender)
    return {
        "session_id": session_id,
        "overall_score": 6.3,
        "potential_score": 7.7,
        "gender": gender,
        "age": age,
        "mp_available": False,
        "zones": zones,
        "daily_plan": _build_daily_plan(zones, gender),
        "priorities": _build_priorities(scores),
        "summary": _build_summary(6.3, 7.7, _build_priorities(scores), gender),
        "message": f"Демо-режим. Ошибка анализа: {error}" if error else "Демо-режим",
    }
