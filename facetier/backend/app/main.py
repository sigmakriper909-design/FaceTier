from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
import tempfile

app = FastAPI(title="FaceTier API", version="0.4.0")

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
    return {"status": "ok", "service": "FaceTier API", "version": "0.4.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug")
async def debug():
    info = {"version": "0.4.0"}
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


def _level(score: float) -> str:
    if score >= 8.0:
        return "strong"
    if score >= 6.5:
        return "ok"
    if score >= 5.0:
        return "weak"
    return "critical"


def _zone_content(zone_id: str, score: float, gender: str) -> dict:
    catalog = {
        "eyes": {
            "name": "Глаза",
            "strong": {
                "description": "Сильная зона. Глаза уже работают на тебя — сохраняй и не порти.",
                "daily": ["Сон 7–9 часов — отёки и тёмные круги убивают даже хороший canthal.", "Холодный компресс 1–2 минуты утром при отёках.", "Не тереть глаза, не спать лицом в подушку."],
                "weekly": ["Проверь, не пересушивает ли кожу вокруг глаз твой уход.", "Раз в неделю — лёгкий лимфодренаж от внутреннего уголка к вискам."],
            },
            "ok": {
                "description": "Нормальная база. Можно визуально усилить взгляд без радикальных мер.",
                "daily": ["Сон 8 часов. Недосып сразу делает взгляд тяжёлым.", "Утром холод на область глаз 60–90 секунд.", "Следи за положением бровей — слишком низкая бровь закрывает глаз."],
                "weekly": ["Форма бровей: чуть приподними хвостик (без женской дуги, если парень).", "Правило 20-20-20 у экрана, чтобы глаза не краснели."],
            },
            "weak": {
                "description": "Зона тянет общую оценку вниз. Фокус на сне, отёках и форме бровей.",
                "daily": ["Жёсткий режим сна: одно время подъёма, 8 часов.", "Меньше соли вечером + 2–2.5 л воды днём.", "Холод на глаза каждое утро.", "Не сутулься у экрана."],
                "weekly": ["Пересмотри форму бровей: убери лишнее снизу у хвостика.", "Если тёмные круги стойкие — сначала сон, не только консилер.", "Оцени canthal: иногда помогает угол брови и причёска."],
            },
            "critical": {
                "description": "Слабое место. Без дисциплины по сну и отёкам прогресса не будет.",
                "daily": ["Сон — приоритет №1.", "Ноль алкоголя и минимум соли после 18:00.", "Холод + лёгкий массаж от центра к вискам каждое утро.", "Держи голову выше при сне."],
                "weekly": ["Фото анфас каждую неделю при одном освещении.", "Брови и причёска: убери всё, что опускает внешний угол глаза.", "Если после 2–3 месяцев дисциплины ноль сдвига — думай про консультацию, не про чудо-кремы."],
            },
        },
        "midface": {
            "name": "Midface",
            "strong": {
                "description": "Хорошая поддержка средней трети. Держи объём и не проседай от сутулости.",
                "daily": ["Mewing: язык на нёбе, губы сомкнуты, дыхание носом.", "Ровная осанка — midface визуально падает, если шея вперёд."],
                "weekly": ["Раз в неделю оцени профиль в зеркале при нейтральном свете."],
            },
            "ok": {
                "description": "Средний уровень. Можно улучшить осанкой, mewing и жиром на лице.",
                "daily": ["Mewing весь день.", "Жуй жвачку или твёрдую пищу осознанно.", "Осанка: уши над плечами."],
                "weekly": ["Если есть лишний вес — дефицит. Лицо часто открывается раньше тела.", "Больше сна на спине, если можешь."],
            },
            "weak": {
                "description": "Средняя треть слабая. Нужна дисциплина: язык, осанка, процент жира.",
                "daily": ["Mewing 24/7 в фоне.", "Жевание 10–15 минут жвачки без сахара после еды.", "Телефон на уровне глаз."],
                "weekly": ["Контроль веса: даже −2–4 кг часто меняют midface.", "Фото профиля раз в неделю."],
            },
            "critical": {
                "description": "Одна из главных зон на прокачку. Без mewing + жира/осанки будет потолок.",
                "daily": ["Mewing постоянно. Напоминание 3 раза в день 2 недели.", "Дыши носом, рот закрыт в покое.", "Осанка + взгляд вперёд при ходьбе."],
                "weekly": ["Дефицит калорий, если жир на лице есть.", "Не жди чуда от упражнений для скул по 5 минут — важнее база."],
            },
        },
        "jaw": {
            "name": "Челюсть",
            "strong": {
                "description": "Сильная челюстная линия. Сохраняй процент жира и тонус.",
                "daily": ["Не наедай лицо вечером — соль и переедание размывают угол.", "Mewing + сомкнутые зубы в покое (не стискивай до боли)."],
                "weekly": ["Раз в неделю проверяй угол челюсти на фото в одном свете."],
            },
            "ok": {
                "description": "База есть. Можно сделать угол заметнее через жир, жевание и осанку.",
                "daily": ["Жвачка 10–15 мин после еды без сахара.", "Mewing.", "Меньше соли и алкоголя."],
                "weekly": ["Силовые + дефицит, если есть лишний вес.", "Не спи лицом в подушку."],
            },
            "weak": {
                "description": "Челюсть — приоритет. Здесь часто самый большой потенциал.",
                "daily": ["Жевание: жвачка или твёрдая пища.", "Mewing + язык на нёбе.", "Вода днём, минимум соли вечером.", "Осанка: подбородок не втягивай в шею."],
                "weekly": ["Контроль калорий. Лишний вес на лице = размытый угол.", "Фото анфас + 45° раз в неделю.", "Борода может усилить линию, но не маскируй ей лень."],
            },
            "critical": {
                "description": "Слабая зона. Без дисциплины по жиру и mewing прогресса не будет.",
                "daily": ["Дефицит / контроль питания, если жир есть.", "Mewing + жевание.", "Ноль отёков: сон, соль, алкоголь."],
                "weekly": ["Тренировки ног/спины/тяги — осанка и общий вид жёстче.", "Не жевательные тренажёры по 2 часа — риск сустава."],
            },
        },
        "cheekbones": {
            "name": "Скулы",
            "strong": {
                "description": "Скулы читаются. Держи жир и не заёбывай отёками.",
                "daily": ["Сон и вода — чтобы скулы не тонули в отёке.", "Осанка."],
                "weekly": ["При смене веса не уходи в слишком низкий жир, если лицо становится измождённым."],
            },
            "ok": {
                "description": "Норм. Можно сделать скулы заметнее через жир и отёки.",
                "daily": ["Контроль соли и сна.", "Не раздувай лицо перееданием на ночь."],
                "weekly": ["При сушке лица скулы часто выходят сами.", "Причёска: не закрывай скулы бесформенной чёлкой."],
            },
            "weak": {
                "description": "Скулы слабо читаются. Работа через процент жира и отёки.",
                "daily": ["Вода, сон, меньше соли.", "Mewing — косвенно помогает средней трети."],
                "weekly": ["Дефицит калорий при лишнем весе.", "Контуринг не заменяет вес и отёк под контролем."],
            },
            "critical": {
                "description": "Скулы почти не держат форму. Приоритет — жир и отёки.",
                "daily": ["Жёсткий контроль вечерней еды и соли.", "Сон 8 часов."],
                "weekly": ["Сушка лица через общий процент жира.", "Не жди эффекта от массажёров без дефицита."],
            },
        },
        "chin": {
            "name": "Подбородок",
            "strong": {
                "description": "Подбородок не слабое звено. Не убивай его осанкой и жиром.",
                "daily": ["Осанка: уши над плечами.", "Язык на нёбе, рот закрыт."],
                "weekly": ["Если появляется подушка под подбородком — калории и подушка."],
            },
            "ok": {
                "description": "Средне. Можно улучшить осанкой, mewing и процентом жира.",
                "daily": ["Mewing.", "Не сиди с головой вперёд в телефон."],
                "weekly": ["Фото профиля — линия от губы к шее."],
            },
            "weak": {
                "description": "Подбородок / переход к шее слабый. Часто лечится жиром + осанкой.",
                "daily": ["Осанка + mewing.", "Не наедай ночь."],
                "weekly": ["Дефицит при лишнем весе.", "Контур бороды может визуально удлинить подбородок."],
            },
            "critical": {
                "description": "Слабое место в профиле. Без контроля жира и осанки будет плыть.",
                "daily": ["Спи на спине, если можешь.", "Mewing + осанка."],
                "weekly": ["Приоритет — вес и сон. Не упражнения для второго подбородка по роликам."],
            },
        },
        "symmetry": {
            "name": "Симметрия",
            "strong": {
                "description": "Симметрия хорошая. Не ломай её привычками.",
                "daily": ["Жуй на обе стороны.", "Не всегда спи на одном боку."],
                "weekly": ["Раз в неделю анфас в зеркало при ровном свете."],
            },
            "ok": {
                "description": "Небольшая асимметрия — норма. Не усугубляй привычками.",
                "daily": ["Жевание на обе стороны.", "Не зажимай голову всегда в одну сторону у телефона."],
                "weekly": ["Фото анфас — отмечай, не растёт ли перекос."],
            },
            "weak": {
                "description": "Асимметрия заметна. Кость так просто не выровнять, но не усугубляй.",
                "daily": ["Жуй равномерно.", "Сон: чередуй стороны или спи на спине."],
                "weekly": ["Причёска и свет маскируют перекос сильнее упражнений на симметрию."],
            },
            "critical": {
                "description": "Сильный перекос. Честные ожидания: скелет так просто не чинится.",
                "daily": ["Не жуй только одной стороной.", "Осанка и сон на спине — минимум."],
                "weekly": ["Если перекос + боль в челюсти — к специалисту, не к гуру mewing."],
            },
        },
        "harmony": {
            "name": "Гармония",
            "strong": {
                "description": "Части лица хорошо стыкуются. Держи сон, вес, кожу.",
                "daily": ["База: сон, вода, чистое лицо утром и вечером.", "Не убивай кожу постоянным фастфудом."],
                "weekly": ["Один честный фото-сет при одном свете."],
            },
            "ok": {
                "description": "В целом складывается. Улучшение слабых зон подтянет гармонию.",
                "daily": ["Сон + mewing + осанка — это и есть гармония в быту."],
                "weekly": ["Работай по приоритетам: не распыляйся на 10 зон сразу."],
            },
            "weak": {
                "description": "Лицо не собирается в одну картинку. Бей в 1–2 слабые зоны.",
                "daily": ["Топ-2 зоны из приоритетов — делай по ним базу каждый день.", "Сон 8 часов."],
                "weekly": ["Не меняй причёску каждую неделю — сначала цифры и дисциплина."],
            },
            "critical": {
                "description": "Сильный разброс по зонам. Нужен фокус, а не 15 привычек сразу.",
                "daily": ["Только база: сон, вода, mewing, осанка.", "Один главный приоритет на 30 дней."],
                "weekly": ["Фото до/после раз в 2 недели. Без этого мозг врёт про прогресс."],
            },
        },
    }
    level = _level(score)
    block = catalog.get(zone_id, catalog["harmony"]).get(level, catalog["harmony"]["ok"])
    return {"description": block["description"], "daily": list(block["daily"]), "weekly": list(block["weekly"]), "level": level, "name": catalog.get(zone_id, {}).get("name", zone_id)}


def _daily_plan(scores: dict, gender: str) -> dict:
    base = [
        "Сон 7–9 часов (одно время подъёма).",
        "Mewing: язык на нёбе, рот закрыт, дыхание носом.",
        "Вода 2–2.5 л днём, минимум соли и алкоголя вечером.",
        "Осанка: телефон на уровне глаз, не шея вперёд.",
        "Умывание утром и вечером + простой увлажняющий крем.",
    ]
    ordered = sorted(scores.items(), key=lambda x: x[1])
    focus = []
    for zone_id, sc in ordered[:2]:
        content = _zone_content(zone_id, sc, gender)
        if content["daily"]:
            focus.append(f"{content.get('name', zone_id)}: {content['daily'][0]}")
    return {"base": base, "focus_today": focus or ["Держи базу 30 дней без пропусков."], "note": "Сначала дисциплина 3–4 недели. Потом смотри цифры снова при том же свете."}


def _summary(overall: float, potential: float, scores: dict) -> str:
    gap = round(potential - overall, 1)
    worst = sorted(scores.items(), key=lambda x: x[1])[:2]
    names = {"eyes": "глаза", "midface": "midface", "jaw": "челюсть", "cheekbones": "скулы", "chin": "подбородок", "symmetry": "симметрия", "harmony": "гармония"}
    wtxt = ", ".join(names.get(z, z) for z, _ in worst)
    if overall >= 7.5:
        return f"Сильная база ({overall}/10). Потенциал ещё +{gap}. Держи сон и процент жира. Слабее: {wtxt}."
    if overall >= 6.0:
        return f"Средний уровень ({overall}/10). Реальный запас +{gap} при дисциплине. Бей в: {wtxt}."
    return f"Сейчас {overall}/10. Потенциал {potential}/10. Фокус: {wtxt}. Без базы (сон/mewing/вес) цифры не сдвинутся."


def _build_full_zones(scores: dict, gender: str) -> list:
    order = ["eyes", "midface", "jaw", "cheekbones", "chin", "symmetry", "harmony"]
    names = {"eyes": "Глаза", "midface": "Midface", "jaw": "Челюсть", "cheekbones": "Скулы", "chin": "Подбородок", "symmetry": "Симметрия", "harmony": "Гармония"}
    zones = []
    for zid in order:
        sc = float(scores.get(zid, 6.0))
        pot = min(9.5, round(sc + (1.6 if sc < 6 else 1.1), 1))
        content = _zone_content(zid, sc, gender)
        zones.append({"id": zid, "name": names[zid], "score": sc, "potential": pot, "level": content["level"], "description": content["description"], "daily": content["daily"], "weekly": content["weekly"]})
    return zones


@app.post("/api/analyze")
async def analyze_face(front: UploadFile = File(...), profile: UploadFile = File(...), gender: str = Form(...), age: int = Form(None)):
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
            mp_ok = analysis.get("mp_available", False)
            warning = analysis.get("warning")
            display_scores = {k: v for k, v in scores.items() if k != "canthal_tilt"} or scores
            canthal_score = float(scores.get("canthal_tilt", scores.get("eyes", 6.5)))
            canthal_value = float(metrics.get("canthal_tilt", 0))
            left_c = metrics.get("left_canthal", 0)
            right_c = metrics.get("right_canthal", 0)
            zones = _build_full_zones(display_scores, gender)
            daily_plan = _daily_plan(display_scores, gender)
            priorities = _build_priorities(display_scores)
            summary = _summary(overall, potential, display_scores)
            msg = "Полный разбор через MediaPipe" if mp_ok else f"Демо-режим: {warning or 'MediaPipe недоступен'}"
            result = {
                "session_id": session_id,
                "overall_score": overall,
                "potential_score": potential,
                "gender": gender,
                "age": age,
                "mp_available": mp_ok,
                "metrics": metrics,
                "summary": summary,
                "daily_plan": daily_plan,
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


def _canthal_description(value: float) -> str:
    if value >= 5:
        return "Сильный положительный кантальный наклон. Глаза выглядят хищными и открытыми."
    if value >= 2:
        return "Положительный кантальный наклон. Хороший показатель."
    if value >= -1:
        return "Нейтральный / слабо положительный наклон. Средний показатель."
    return "Отрицательный кантальный наклон. Глаза выглядят более усталыми."


def _canthal_advice(value: float) -> list:
    if value >= 4:
        return ["Отличный показатель. Поддерживай сном и формой бровей.", "Избегай отёков."]
    if value >= 1:
        return ["Хороший уровень. Форма бровей может усилить взгляд.", "Следи за сном и кругами под глазами."]
    return ["Есть потенциал. Брови и нижнее веко влияют сильнее, чем кажется.", "Сначала дисциплина 2–3 месяца."]


def _build_priorities(scores: dict) -> list:
    names = {"jaw": "Челюсть", "midface": "Midface", "harmony": "Гармония", "cheekbones": "Скулы", "eyes": "Глаза", "chin": "Подбородок", "symmetry": "Симметрия"}
    items = [(names.get(k, k), v) for k, v in scores.items()]
    items.sort(key=lambda x: x[1])
    return [f"{name} — наибольший потенциал улучшения" for name, _ in items[:3]]


def _demo_result(session_id, gender, age, error=None):
    scores = {"eyes": 6.5, "midface": 6.0, "jaw": 5.8, "cheekbones": 6.3, "chin": 6.0, "symmetry": 6.8, "harmony": 6.2}
    zones = _build_full_zones(scores, gender or "male")
    return {
        "session_id": session_id,
        "overall_score": 6.3,
        "potential_score": 7.7,
        "gender": gender,
        "age": age,
        "mp_available": False,
        "summary": _summary(6.3, 7.7, scores),
        "daily_plan": _daily_plan(scores, gender or "male"),
        "demo_parameter": {"name": "Canthal Tilt", "name_ru": "Кантальный наклон", "score": 6.5, "potential": 8.0, "value": "+2.1°", "description": "Демо-режим.", "advice": ["Демо."],},
        "zones": zones,
        "priorities": _build_priorities(scores),
        "message": f"Демо-режим. Ошибка: {error}" if error else "Демо-режим",
    }
