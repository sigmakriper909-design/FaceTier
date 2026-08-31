from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import tempfile

app = FastAPI(title="FaceTier API", version="0.6.0")

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
    return {"status": "ok", "service": "FaceTier API", "version": "0.6.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug")
async def debug():
    info = {"version": "0.6.0"}
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
                str(front_path), str(profile_path), gender=gender, age=age,
            )
            scores = analysis.get("scores", {})
            overall = analysis.get("overall_score", 6.0)
            potential = analysis.get("potential_score", 7.5)
            metrics = analysis.get("metrics", {})
            mp_ok = analysis.get("mp_available", False)
            warning = analysis.get("warning")

            zones = _build_full_zones(scores, metrics, gender)
            daily = _build_daily_plan(zones, gender)
            priorities = _build_priorities(scores)

            msg = (
                "Полный разбор выполнен через MediaPipe"
                if mp_ok
                else f"Демо-режим: {warning or 'MediaPipe недоступен'}"
            )

            for z in zones:
                zid = z.get("id")
                z["photo"] = "profile" if zid in ("jaw", "chin", "midface") else "front"
                z["overlay_key"] = "eyes" if zid == "eyes" else zid

            result = {
                "session_id": session_id,
                "overall_score": overall,
                "potential_score": potential,
                "gender": gender,
                "age": age,
                "mp_available": mp_ok,
                "metrics": metrics,
                "overlays": analysis.get("overlays") or {},
                "profile_mesh": analysis.get("profile_mesh", False),
                "zones": zones,
                "daily_plan": daily,
                "priorities": priorities,
                "summary": _build_summary(overall, potential, priorities, gender),
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


def _level(score: float) -> str:
    if score >= 8.0:
        return "strong"
    if score >= 6.5:
        return "ok"
    if score >= 5.0:
        return "weak"
    return "critical"


def _build_full_zones(scores: dict, metrics: dict, gender: str) -> list:
    canthal = float(metrics.get("canthal_tilt", 0))
    midface_r = float(metrics.get("midface_ratio", 1.0))
    jaw_r = float(metrics.get("jaw_to_cheek", 0.85))
    sym_err = float(metrics.get("symmetry_error", 5.0))

    def zone(zid, name, score_key, potential_extra, desc_fn, daily_fn, weekly_fn):
        sc = float(scores.get(score_key, 6.0))
        pot = min(9.5, sc + potential_extra)
        return {
            "id": zid, "name": name, "score": round(sc, 1), "potential": round(pot, 1),
            "level": _level(sc), "description": desc_fn(sc),
            "daily": daily_fn(sc), "weekly": weekly_fn(sc),
        }

    eyes_desc = lambda s: (
        f"Canthal tilt {canthal:+.1f}. "
        + ("Сильный положительный наклон — hunter eyes vibe." if canthal >= 4
           else "Положительный наклон, глаза выглядят живее." if canthal >= 1.5
           else "Нейтральный/слабый наклон — взгляд может казаться более усталым." if canthal >= -1
           else "Отрицательный canthal tilt — глаза визуально тяжелее и закрытее.")
    )
    eyes_daily = lambda s: (
        ["Сон 7–9 часов — отёки убивают взгляд сильнее, чем ты думаешь",
         "Холодный компресс / охлаждённые ложки 2–3 мин утром под глаза",
         "Не тереть глаза, не спать лицом в подушку",
         "Форма бровей: чуть приподнятый хвост визуально усиливает canthal"]
        if s < 7 else
        ["Держи режим сна — это твоё главное преимущество",
         "Лёгкий холод утром, если есть склонность к отёкам",
         "Не перегружай нижнее веко косметикой/фильтрами"]
    )
    eyes_weekly = lambda s: (
        ["Раз в неделю — маска/патчи под глаза (кофеин или пептиды)",
         "Проверь, не опускаются ли брови (это визуально портит tilt)"]
        if s < 7.5 else ["Поддерживающий уход под глаза 1–2 раза в неделю"]
    )

    mid_desc = lambda s: (
        f"Соотношение midface {midface_r:.2f}. "
        + ("Компактный midface — сильный показатель." if 0.95 <= midface_r <= 1.2
           else "Midface чуть вытянут — лицо может казаться длиннее." if midface_r > 1.2
           else "Midface короткий/сжатый — следи за гармонией с нижней третью.")
    )
    mid_daily = lambda s: [
        "Осанка: подбородок чуть назад и вниз, шея длинная — меняет восприятие midface",
        "Жевание (жвачка / жевательный тренинг) 10–15 мин — тонус masseter",
        "Не запрокидывай голову на фото и в зеркале — это искажает midface",
        "Дыши носом, язык к нёбу (mewing) — базовый daily",
    ]
    mid_weekly = lambda s: [
        "Сравни фото анфас раз в 2 недели при одинаковом свете",
        "Если lower third слабый — приоритет челюсти/подбородку, не тянуть midface отдельно",
    ]

    jaw_desc = lambda s: (
        f"Соотношение челюсть/скулы ~{jaw_r:.2f}. "
        + ("Сильная, читаемая челюсть." if s >= 7.5
           else "Средняя челюсть — есть запас по углу и ширине." if s >= 5.5
           else "Челюсть слабая/сглаженная — главный рычаг для общей оценки.")
    )
    jaw_daily = lambda s: [
        "Жевательная нагрузка: жвачка (без сахара) 15–20 мин или жевательный тренинг",
        "Язык к нёбу весь день (mewing) — без силы, просто контакт",
        "Не дыши ртом — рот закрыт в покое",
        "Поза: уши над плечами, подбородок не вперёд",
    ]
    jaw_weekly = lambda s: (
        ["Прогрессия жевательной нагрузки (осторожно, без боли в суставе)",
         "Фото профиля раз в неделю при одном угле и свете",
         "Если жир на лице — дефицит калорий важнее любых упражнений"]
        if s < 7 else ["Поддерживай жевательную нагрузку и lean face"]
    )

    cheek_desc = lambda s: (
        "Скулы читаются, есть структура." if s >= 7
        else "Скулы средние — можно усилить через lean face и свет." if s >= 5.5
        else "Скулы слабо выражены — чаще всего помогает снижение подкожного жира + угол света."
    )
    cheek_daily = lambda s: [
        "Lean face: если есть лишний жир — держи лёгкий дефицит",
        "Соль и алкоголь вечером = отеки и плоские скулы утром",
        "Пей воду равномерно, не залпом на ночь",
        "Причёска/объём сверху не должен спорить со скулами",
    ]
    cheek_weekly = lambda s: [
        "Контроль веса и фото при одном освещении",
        "Не гонись за западанием щёк через экстремальный дефицит — будет выглядеть хуже",
    ]

    chin_desc = lambda s: (
        "Подбородок держит нижнюю треть." if s >= 7
        else "Подбородок средний — следи, чтобы не терялся в шее." if s >= 5.5
        else "Слабый/скошенный подбородок сильно бьёт по профилю и гармонии."
    )
    chin_daily = lambda s: [
        "Осанка и mewing — подбородок не должен уезжать вперёд от привычки",
        "Не прижимай подбородок к шее в фото и селфи",
        "Если двойной подбородок — lean + осанка важнее упражнений",
    ]
    chin_weekly = lambda s: [
        "Фото строгого профиля раз в неделю",
        "При сильном рецессе подбородка долгосрочно рассматривают импланты/гениопластику — это уже не daily",
    ]

    sym_desc = lambda s: (
        f"Ошибка симметрии (грубо) {sym_err:.1f}. "
        + ("Симметрия хорошая." if s >= 7.5
           else "Есть заметная асимметрия — почти у всех, но её можно визуально сгладить." if s >= 5.5
           else "Асимметрия сильная — часть из этого привычки (жевание на одну сторону, сон).")
    )
    sym_daily = lambda s: [
        "Жуй равномерно обеими сторонами",
        "Не спи всегда на одном боку лицом в подушку",
        "Проверь, не поднимаешь ли одну бровь/уголок рта привычкой",
    ]
    sym_weekly = lambda s: [
        "Сравни зеркальные фото (flip) раз в 2 недели",
        "Сильную скелетную асимметрию daily не исправит — только визуал и привычки",
    ]

    harm_desc = lambda s: (
        "Части лица согласованы между собой." if s >= 7.5
        else "Гармония средняя — отдельные зоны тянут вниз общую оценку." if s >= 5.5
        else "Дисбаланс зон: сначала закрывай самые слабые места, не всё сразу."
    )
    harm_daily = lambda s: [
        "Один главный фокус на день из приоритетов ниже — не распыляйся",
        "Сон, вода, осанка, закрытый рот в покое — база для всего лица",
        "Одинаковый свет и ракурс в контрольных фото",
    ]
    harm_weekly = lambda s: [
        "Раз в неделю: 3 фото (анфас, 45°, профиль) при одном свете",
        "Сверяйся с приоритетами — сначала weak/critical зоны",
    ]

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
    base = [
        "Сон 7–9 часов (без этого лицо плывёт)",
        "Рот закрыт в покое, дыхание носом, язык к нёбу",
        "Осанка: уши над плечами, подбородок не выпирает вперёд",
        "Вода в течение дня, меньше соли и алкоголя вечером",
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
        "Сильная база. Дальше — точечная доводка и дисциплина." if overall >= 7.5
        else "Средний уровень. Рост реально упирается в 2–3 слабые зоны и daily-привычки." if overall >= 5.5
        else "Сейчас лицо тянут вниз слабые зоны. Не распыляйся: база + топ-приоритеты."
    )
    top = priorities[0] if priorities else "дисциплина"
    return (
        f"Оценка {overall:.1f}/10, потенциал около {potential:.1f} (запас ~{gap:.1f}). "
        f"{tone} Ближайший фокус: {top}."
    )


def _demo_result(session_id, gender, age, error=None):
    scores = {
        "canthal_tilt": 6.5, "eyes": 6.5, "midface": 6.0, "jaw": 5.8,
        "cheekbones": 6.3, "chin": 6.0, "symmetry": 6.8, "harmony": 6.2,
    }
    zones = _build_full_zones(scores, {"canthal_tilt": 0, "midface_ratio": 1.0, "jaw_to_cheek": 0.85, "symmetry_error": 5}, gender)
    return {
        "session_id": session_id, "overall_score": 6.3, "potential_score": 7.7,
        "gender": gender, "age": age, "mp_available": False,
        "zones": zones, "daily_plan": _build_daily_plan(zones, gender),
        "priorities": _build_priorities(scores),
        "summary": _build_summary(6.3, 7.7, _build_priorities(scores), gender),
        "message": f"Демо-режим. Ошибка анализа: {error}" if error else "Демо-режим",
        "overlays": {}, "profile_mesh": False,
    }
