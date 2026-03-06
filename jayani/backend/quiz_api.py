

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from bson import ObjectId
from collections import Counter, defaultdict

from dbconnect import col
from auth import get_current_user

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


# ---------------------------- Models ----------------------------
class QuizStartIn(BaseModel):
    questions: list[dict] = Field(default_factory=list)
    mode: str = "voice"


class QuizStartOut(BaseModel):
    session_id: str
    count: int


class QuizSubmitIn(BaseModel):
    session_id: str
    answers: list[dict] = Field(default_factory=list)
    total_marks: int = 0  # ignored; we recompute


# ---------------------------- Helpers ----------------------------
def _pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100.0, 2)


def _norm_topic(t: str) -> str:
    t = (t or "").strip()
    return t if t else "UNKNOWN"


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _dt_iso(d):
    try:
        if not d:
            return None
        return d.isoformat()
    except Exception:
        return None


def _ensure_topics_in_answers(answers: list[dict], questions: list[dict]) -> list[dict]:
    qmap = {}
    for q in (questions or []):
        qid = str(q.get("question_id") or "").strip()
        if not qid:
            continue
        qmap[qid] = _norm_topic(q.get("topic") or q.get("Topics") or "")

    fixed = []
    for a in (answers or []):
        if not isinstance(a, dict):
            continue
        qid = str(a.get("question_id") or "").strip()
        topic = a.get("topic")
        if not topic:
            topic = qmap.get(qid, "UNKNOWN")
        a2 = dict(a)
        a2["topic"] = _norm_topic(topic)
        fixed.append(a2)
    return fixed


# ---------------------------- Report builders ----------------------------
def build_report_and_voice_pattern(answers: list[dict]) -> tuple[dict, dict, int]:
    n = len(answers)

    total_marks = 0
    grade_levels = []
    no_speech_count = 0
    voice_labels = []

    for a in answers:
        grade = a.get("grade") or {}
        marks = grade.get("marks") or 0
        level = grade.get("level") or "N/A"

        total_marks += _safe_int(marks, 0)

        if isinstance(level, str) and level.strip():
            grade_levels.append(level.strip().upper())

        vc = a.get("voice_confidence") or {}
        label = vc.get("predicted_label")

        if isinstance(label, str) and label.strip():
            lab = label.strip()
            if lab == "NO_SPEECH":
                no_speech_count += 1
            elif lab != "N/A":
                voice_labels.append(lab)

    grade_counter = Counter(grade_levels)
    good = int(grade_counter.get("GOOD", 0))
    partial = int(grade_counter.get("PARTIAL", 0))
    weak = int(grade_counter.get("WEAK", 0))
    incorrect = int(grade_counter.get("INCORRECT", 0))

    max_marks = n * 10

    report_summary = {
        "total_questions": n,
        "max_marks": max_marks,
        "total_marks": total_marks,
        "levels": {
            "GOOD": {"count": good, "percent": _pct(good, n)},
            "PARTIAL": {"count": partial, "percent": _pct(partial, n)},
            "WEAK": {"count": weak, "percent": _pct(weak, n)},
            "INCORRECT": {"count": incorrect, "percent": _pct(incorrect, n)},
        },
        "no_speech": {"count": no_speech_count, "percent": _pct(no_speech_count, n)},
    }

    voice_counter = Counter(voice_labels)
    total_voice = sum(voice_counter.values())

    voice_pattern = {
        "total_voice_answers": int(total_voice),
        "labels": [],
        "overall_label": None,
    }

    if total_voice > 0:
        for lab, cnt in voice_counter.most_common():
            voice_pattern["labels"].append(
                {"label": str(lab), "count": int(cnt), "percent": _pct(int(cnt), total_voice)}
            )
        voice_pattern["overall_label"] = voice_counter.most_common(1)[0][0]

    return report_summary, voice_pattern, total_marks


def build_topic_report(answers: list[dict]) -> dict:
    buckets = defaultdict(list)
    for a in answers:
        buckets[_norm_topic(a.get("topic"))].append(a)

    topic_rows = []
    for topic, items in buckets.items():
        n = len(items)
        marks_list = []
        lvl_counter = Counter()

        for it in items:
            grade = it.get("grade") or {}
            marks = _safe_int(grade.get("marks"), 0)
            level = str(grade.get("level") or "INCORRECT").strip().upper()
            if level not in ("GOOD", "PARTIAL", "WEAK", "INCORRECT"):
                level = "INCORRECT"

            marks_list.append(marks)
            lvl_counter[level] += 1

        avg_marks = round(sum(marks_list) / max(1, len(marks_list)), 2)
        incorrect_pct = _pct(int(lvl_counter.get("INCORRECT", 0)), n)

        topic_rows.append(
            {
                "topic": topic,
                "count": n,
                "avg_marks": avg_marks,
                "incorrect_percent": incorrect_pct,
                "levels": {
                    "GOOD": {"count": int(lvl_counter.get("GOOD", 0)), "percent": _pct(int(lvl_counter.get("GOOD", 0)), n)},
                    "PARTIAL": {"count": int(lvl_counter.get("PARTIAL", 0)), "percent": _pct(int(lvl_counter.get("PARTIAL", 0)), n)},
                    "WEAK": {"count": int(lvl_counter.get("WEAK", 0)), "percent": _pct(int(lvl_counter.get("WEAK", 0)), n)},
                    "INCORRECT": {"count": int(lvl_counter.get("INCORRECT", 0)), "percent": _pct(int(lvl_counter.get("INCORRECT", 0)), n)},
                },
            }
        )

    best_topic = None
    worst_topic = None
    if topic_rows:
        best_topic = sorted(topic_rows, key=lambda r: (r["avg_marks"], r["count"]), reverse=True)[0]["topic"]
        worst_topic = sorted(topic_rows, key=lambda r: (r["avg_marks"], -r["incorrect_percent"], r["count"]))[0]["topic"]

    return {
        "topics": sorted(topic_rows, key=lambda r: (r["avg_marks"], r["count"]), reverse=True),
        "best_topic": best_topic,
        "worst_topic": worst_topic,
    }


def _topic_subset(answers: list[dict], topic: str) -> list[dict]:
    t = _norm_topic(topic)
    return [a for a in answers if _norm_topic(a.get("topic")) == t]


def _voice_overall_from_answers(answers: list[dict]) -> str:
    labs = []
    for a in answers:
        vc = a.get("voice_confidence") or {}
        lab = vc.get("predicted_label")
        if isinstance(lab, str):
            lab = lab.strip()
            if lab and lab not in ("NO_SPEECH", "N/A"):
                labs.append(lab)
    if not labs:
        return "N/A"
    return Counter(labs).most_common(1)[0][0]


def build_feedback(answers: list[dict], voice_pattern: dict, topic_report: dict) -> dict:
    worst_topic = (topic_report or {}).get("worst_topic") or "UNKNOWN"
    best_topic = (topic_report or {}).get("best_topic") or "UNKNOWN"

    worst_items = _topic_subset(answers, worst_topic)
    worst_voice = _voice_overall_from_answers(worst_items) if worst_items else None

    overall_voice = (voice_pattern or {}).get("overall_label") or _voice_overall_from_answers(answers) or "N/A"
    voice_for_feedback = worst_voice or overall_voice or "N/A"

    worst_avg = None
    worst_incorrect_pct = None
    for t in (topic_report or {}).get("topics") or []:
        if _norm_topic(t.get("topic")) == _norm_topic(worst_topic):
            worst_avg = _safe_float(t.get("avg_marks"), None)
            worst_incorrect_pct = _safe_float(t.get("incorrect_percent"), None)
            break

    is_misconception = False
    if str(voice_for_feedback).strip().lower() == "confident":
        if worst_avg is not None and worst_avg <= 4.0:
            is_misconception = True
        if worst_incorrect_pct is not None and worst_incorrect_pct >= 50.0:
            is_misconception = True

    vf = str(voice_for_feedback).strip().lower()
    if vf == "nervous":
        main_msg = (
            f"You are struggling with {worst_topic}. We detected high jitter (nervousness) in your voice. "
            f"Please review the basic definitions of {worst_topic}."
        )
    elif vf == "hesitant":
        main_msg = (
            f"You scored low in {worst_topic}. You were hesitant and took long pauses. "
            f"Practice more questions on {worst_topic} to build confidence."
        )
    elif vf == "confident":
        if is_misconception:
            main_msg = (
                f"CRITICAL ALERT: You spoke very confidently in {worst_topic}, but your answers were wrong. "
                f"This indicates a Misconception. Please re-study {worst_topic}."
            )
        else:
            main_msg = (
                f"You sounded confident overall. Still, your weakest area is {worst_topic}. "
                f"Revise {worst_topic} and attempt more questions."
            )
    else:
        main_msg = f"Your weakest topic is {worst_topic}. Practice more questions in {worst_topic}."

    best_msg = (
        f"Your strongest topic is {best_topic}. Keep it up."
        if best_topic and best_topic != "UNKNOWN"
        else "Good effort. Keep practicing."
    )

    return {
        "overall_voice": overall_voice,
        "voice_for_feedback": voice_for_feedback,
        "best_topic": best_topic,
        "worst_topic": worst_topic,
        "misconception": bool(is_misconception),
        "messages": {"main": main_msg, "best_topic": best_msg},
    }


# ---------------------------- Topic Insights (bars + scatter) ----------------------------
def _confidence_y(label: str) -> float:
    lab = (label or "").strip().lower()
    if lab == "confident":
        return 0.8
    if lab == "hesitant":
        return 0.5
    if lab == "nervous":
        return 0.25
    return 0.0


def build_topic_insights(answers: list[dict], topic_report: dict) -> dict:
    topics = (topic_report or {}).get("topics") or []
    topic_bars = []
    scatter_points = []

    for trow in topics:
        topic = _norm_topic(trow.get("topic"))
        avg = _safe_float(trow.get("avg_marks"), 0.0)
        incorrect_pct = _safe_float(trow.get("incorrect_percent"), 0.0)

        percent = round((avg / 10.0) * 100.0, 2)

        status = "Needs Work"
        if percent >= 85:
            status = "Mastery"
        elif percent >= 60:
            status = "Good"
        elif percent >= 40:
            status = "Average"
        else:
            status = "Weak"

        items = _topic_subset(answers, topic)
        topic_voice = _voice_overall_from_answers(items)

        misconception = False
        if (topic_voice or "").strip().lower() == "confident":
            if avg <= 4.0 or incorrect_pct >= 50.0:
                misconception = True

        topic_bars.append(
            {
                "topic": topic,
                "avg_marks": avg,
                "percent": percent,
                "status": status,
                "voice_label": topic_voice,
                "incorrect_percent": incorrect_pct,
                "misconception": bool(misconception),
            }
        )

        scatter_points.append(
            {"topic": topic, "x": avg, "y": _confidence_y(topic_voice), "voice_label": topic_voice}
        )

    return {"topic_bars": topic_bars, "scatter_points": scatter_points}


# ---------------------------- Routes ----------------------------
@router.post("/start", response_model=QuizStartOut)
def start_quiz(body: QuizStartIn, user=Depends(get_current_user)):
    if not body.questions:
        raise HTTPException(status_code=400, detail="questions list required")

    sessions = col("quiz_sessions")

    doc = {
        "user_id": user["id"],
        "user_email": user.get("email"),
        "mode": body.mode,
        "questions": body.questions,
        "answers": [],
        "total_marks": 0,
        "report_summary": None,
        "voice_pattern": None,
        "topic_report": None,
        "feedback": None,
        "topic_insights": None,
        "started_at": datetime.now(timezone.utc),
        "submitted_at": None,
        "status": "in_progress",
    }

    res = sessions.insert_one(doc)
    return {"session_id": str(res.inserted_id), "count": len(body.questions)}


@router.post("/submit")
def submit_quiz(body: QuizSubmitIn, user=Depends(get_current_user)):
    if not body.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    sessions = col("quiz_sessions")
    try:
        _id = ObjectId(body.session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid session_id")

    sess = sessions.find_one({"_id": _id, "user_id": user["id"]})
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    answers_fixed = _ensure_topics_in_answers(body.answers, sess.get("questions") or [])

    report_summary, voice_pattern, total = build_report_and_voice_pattern(answers_fixed)
    topic_report = build_topic_report(answers_fixed)
    feedback = build_feedback(answers_fixed, voice_pattern, topic_report)
    topic_insights = build_topic_insights(answers_fixed, topic_report)

    submitted_at = datetime.now(timezone.utc)

    sessions.update_one(
        {"_id": _id},
        {
            "$set": {
                "answers": answers_fixed,
                "total_marks": total,
                "report_summary": report_summary,
                "voice_pattern": voice_pattern,
                "topic_report": topic_report,
                "feedback": feedback,
                "topic_insights": topic_insights,
                "submitted_at": submitted_at,
                "status": "submitted",
            }
        },
    )

    return {
        "ok": True,
        "session_id": body.session_id,
        "answers": answers_fixed,
        "total_marks": total,
        "report_summary": report_summary,
        "voice_pattern": voice_pattern,
        "topic_report": topic_report,
        "feedback": feedback,
        "topic_insights": topic_insights,
        "submitted_at": submitted_at.isoformat(),
    }


@router.get("/submitted")
def my_submitted_quizzes(user=Depends(get_current_user)):
    sessions = col("quiz_sessions")
    items = []

    cur = (
        sessions.find({"user_id": user["id"], "status": "submitted"})
        .sort("submitted_at", -1)
        .limit(50)
    )

    for d in cur:
        report = d.get("report_summary") or {}
        topic_report = d.get("topic_report") or {}
        feedback = d.get("feedback") or {}
        topic_insights = d.get("topic_insights") or {}

        items.append(
            {
                "session_id": str(d["_id"]),
                "mode": d.get("mode"),
                "total_marks": int(d.get("total_marks", 0)),
                "max_marks": int(report.get("max_marks", 0)),
                "total_questions": int(report.get("total_questions", 0)),
                "started_at": _dt_iso(d.get("started_at")),
                "submitted_at": _dt_iso(d.get("submitted_at")),
                "report_summary": report,
                "best_topic": topic_report.get("best_topic"),
                "worst_topic": topic_report.get("worst_topic"),
                "overall_voice": feedback.get("overall_voice"),
                "topic_bars": topic_insights.get("topic_bars", []),
                "scatter_points": topic_insights.get("scatter_points", []),

                # ✅ feedback only
                "feedback": {
                    "overall_voice": feedback.get("overall_voice"),
                    "voice_for_feedback": feedback.get("voice_for_feedback"),
                    "best_topic": feedback.get("best_topic"),
                    "worst_topic": feedback.get("worst_topic"),
                    "misconception": bool(feedback.get("misconception", False)),
                    "messages": (feedback.get("messages") or {}),
                },
            }
        )

    return {"items": items}


@router.get("/submitted-summary")
def submitted_summary(user=Depends(get_current_user)):
    sessions = col("quiz_sessions")
    cur = sessions.find({"user_id": user["id"], "status": "submitted"}, {"report_summary": 1, "total_marks": 1})

    total_attempts = 0
    total_marks_sum = 0
    max_marks_sum = 0
    lvl_counts = Counter()
    total_questions_sum = 0

    for d in cur:
        total_attempts += 1
        rs = d.get("report_summary") or {}
        total_marks_sum += int(rs.get("total_marks", d.get("total_marks", 0)) or 0)
        max_marks_sum += int(rs.get("max_marks", 0) or 0)
        tq = int(rs.get("total_questions", 0) or 0)
        total_questions_sum += tq

        lvls = (rs.get("levels") or {})
        for k in ("GOOD", "PARTIAL", "WEAK", "INCORRECT"):
            lvl_counts[k] += int((lvls.get(k) or {}).get("count", 0) or 0)

    avg_percent = 0.0
    if max_marks_sum > 0:
        avg_percent = round((total_marks_sum / max_marks_sum) * 100.0, 2)

    levels_percent = {}
    for k in ("GOOD", "PARTIAL", "WEAK", "INCORRECT"):
        levels_percent[k] = {
            "count": int(lvl_counts.get(k, 0)),
            "percent": _pct(int(lvl_counts.get(k, 0)), int(total_questions_sum or 0)),
        }

    return {
        "total_attempts": total_attempts,
        "avg_percent": avg_percent,
        "levels": levels_percent,
        "total_questions": int(total_questions_sum),
    }
