


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py
============================================
ICT Short Answer + Voice API (IDEAL ANSWER WORD-COVERAGE GRADING)

FIXES ADDED ✅
1) Correct model path loading (relative to app.py folder)
2) Silence / No-speech detection -> returns "NO_SPEECH" instead of random Nervous/Hesitant
3) Convert browser audio (webm/ogg/m4a/mp3) -> WAV using ffmpeg (if installed)
4) Better debug info in /health

Run:
  python app.py
"""

import os
import re
import tempfile
from datetime import datetime
import subprocess

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import librosa
import joblib
import difflib


# ---------------------- Base dir (IMPORTANT FIX) ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../Backend
PROJECT_DIR = os.path.dirname(BASE_DIR)                 # project root
FRONTEND_DIR = os.path.join(PROJECT_DIR, "Frontend")    # .../Frontend


# ---------------------- Whisper (Local STT) -----------------------
import whisper
print("[WHISPER] Loading local Whisper model 'base' ...")
WHISPER_MODEL = whisper.load_model("base")
print("[WHISPER] Model loaded.")

# ---------------------- Text Normalization ------------------------
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "at", "from", "into", "about", "than", "then",
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_important(text: str) -> list:
    text = clean_text(text)
    toks = [t for t in text.split() if len(t) >= 2 and t not in STOPWORDS]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def fuzzy_token_match(token: str, transcript_tokens: list, thr: float = 0.82):
    if token in transcript_tokens:
        return True, token

    best = ("", 0.0)
    for w in transcript_tokens:
        if abs(len(w) - len(token)) >= 5:
            continue
        r = difflib.SequenceMatcher(None, token, w).ratio()
        if r > best[1]:
            best = (w, r)

    if best[1] >= thr:
        return True, best[0]
    return False, ""

def ideal_word_coverage(transcript: str, ideal_answer: str, fuzzy_thr: float = 0.82) -> dict:
    ideal_tokens = tokenize_important(ideal_answer)
    trans_tokens = tokenize_important(transcript)

    matched = []
    unmatched = []
    matches_map = {}

    if len(ideal_tokens) == 0:
        return {
            "coverage": 0.0,
            "ideal_tokens": [],
            "transcript_tokens": trans_tokens,
            "matched_tokens": [],
            "unmatched_tokens": [],
            "matches_map": {},
            "note": "ideal_answer_has_no_tokens_after_cleaning",
        }

    for t in ideal_tokens:
        ok, m = fuzzy_token_match(t, trans_tokens, thr=fuzzy_thr)
        if ok:
            matched.append(t)
            matches_map[t] = m
        else:
            unmatched.append(t)

    coverage = len(matched) / len(ideal_tokens)
    return {
        "coverage": float(coverage),
        "ideal_tokens": ideal_tokens,
        "transcript_tokens": trans_tokens,
        "matched_tokens": matched,
        "unmatched_tokens": unmatched,
        "matches_map": matches_map,
        "fuzzy_threshold": float(fuzzy_thr),
    }

def grade_by_ideal_coverage(coverage: float) -> dict:
    if coverage >= 0.75:
        return {"level": "GOOD", "marks": 10, "coverage": float(coverage)}
    if coverage >= 0.50:
        return {"level": "PARTIAL", "marks": 7, "coverage": float(coverage)}
    if coverage >= 0.35:
        return {"level": "WEAK", "marks": 3, "coverage": float(coverage)}
    return {"level": "INCORRECT", "marks": 0, "coverage": float(coverage)}

# ---------------------- Voice confidence model ---------------------
# IMPORTANT FIX: resolve model path relative to app.py
VOICE_MODEL_PATH = os.environ.get(
    "VOICE_MODEL_PATH",
    os.path.join(BASE_DIR, "voice_confidence_model.joblib")
)

VOICE_CLF = None
VOICE_LE = None
VOICE_SAMPLE_RATE = 16000

VOICE_BACKEND = "mfcc"  # "mfcc" or "yamnet"
YAMNET_HANDLE = None
EMBED_POOL = "mean_std"
_YAMNET_MODEL = None

def _load_voice_model():
    global VOICE_CLF, VOICE_LE, VOICE_SAMPLE_RATE, VOICE_BACKEND, YAMNET_HANDLE, EMBED_POOL
    try:
        bundle = joblib.load(VOICE_MODEL_PATH)
        VOICE_CLF = bundle["model"]
        VOICE_LE = bundle["label_encoder"]
        VOICE_SAMPLE_RATE = bundle.get("sample_rate", VOICE_SAMPLE_RATE)

        VOICE_BACKEND = bundle.get("embedding_backend", "mfcc")
        YAMNET_HANDLE = bundle.get("yamnet_handle", None)
        EMBED_POOL = bundle.get("embed_pool", EMBED_POOL)

        print("[VOICE] Loaded model from:", VOICE_MODEL_PATH)
        print("[VOICE] Backend:", VOICE_BACKEND)
        print("[VOICE] Classes:", list(VOICE_LE.classes_))

        if VOICE_BACKEND == "yamnet" and not YAMNET_HANDLE:
            raise RuntimeError("backend=yamnet but 'yamnet_handle' missing in model bundle.")
    except Exception as e:
        print(f"[VOICE] Could not load voice confidence model: {e}")
        VOICE_CLF = None
        VOICE_LE = None

def _get_yamnet():
    global _YAMNET_MODEL
    if _YAMNET_MODEL is None:
        import tensorflow_hub as hub
        print(f"[VOICE] Loading YAMNet from TF Hub: {YAMNET_HANDLE}")
        _YAMNET_MODEL = hub.load(YAMNET_HANDLE)
    return _YAMNET_MODEL

def _extract_yamnet_embedding(path: str):
    try:
        import tensorflow as tf
    except Exception as e:
        print("[VOICE] TensorFlow not available:", e)
        return None

    try:
        y, _sr = librosa.load(path, sr=VOICE_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"[VOICE] Failed to load audio '{path}': {e}")
        return None
    if y.size == 0:
        return None

    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    yamnet = _get_yamnet()
    _scores, embeddings, _spectrogram = yamnet(waveform)

    emb = embeddings.numpy()
    if emb.size == 0:
        return None

    if EMBED_POOL == "mean":
        vec = emb.mean(axis=0)
    elif EMBED_POOL == "mean_std":
        vec = np.concatenate([emb.mean(axis=0), emb.std(axis=0)], axis=0)
    else:
        return None

    return vec.astype(np.float32)

def _extract_mfcc_features(path: str):
    try:
        y, sr = librosa.load(path, sr=VOICE_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"[VOICE] Failed to load audio '{path}': {e}")
        return None
    if y.size == 0:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(zcr.mean())
    zcr_std = float(zcr.std())

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.array(tempo).ravel()[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    except Exception:
        tempo = 0.0

    stats_vec = np.array([zcr_mean, zcr_std, rms_mean, rms_std, tempo], dtype=np.float32)
    feats = np.concatenate([mfcc_mean, mfcc_std, stats_vec]).astype(np.float32)
    return feats

# ---------------------- SILENCE / NO SPEECH FIX ✅ ----------------------
SILENCE_RMS_THRESHOLD = float(os.environ.get("SILENCE_RMS_THRESHOLD", "0.008"))

def _rms_mean(path: str, sr: int = 16000) -> float:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=y)[0]
        return float(rms.mean()) if rms.size else 0.0
    except Exception:
        return 0.0

def _is_silence(path: str) -> bool:
    return _rms_mean(path, sr=VOICE_SAMPLE_RATE) < SILENCE_RMS_THRESHOLD

# ---------------------- FFmpeg convert to wav FIX ✅ ----------------------
def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def ensure_wav(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path
    if not _ffmpeg_exists():
        return path  # fallback

    out_wav = path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(VOICE_SAMPLE_RATE), out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return out_wav
    except Exception as e:
        print("[AUDIO] ffmpeg convert failed:", e)
        return path

def predict_voice_confidence(path: str) -> dict:
    if VOICE_CLF is None or VOICE_LE is None:
        return {"error": "voice_model_not_loaded"}

    # ✅ Prevent silence from becoming Nervous/Hesitant
    if _is_silence(path):
        return {
            "predicted_label": "NO_SPEECH",
            "probabilities": {},
            "note": f"silence_detected_rms<thr({SILENCE_RMS_THRESHOLD})"
        }

    vec = _extract_yamnet_embedding(path) if VOICE_BACKEND == "yamnet" else _extract_mfcc_features(path)
    if vec is None:
        return {"error": "could_not_extract_features"}

    X = vec.reshape(1, -1)

    try:
        if hasattr(VOICE_CLF, "predict_proba"):
            probs = VOICE_CLF.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            label = VOICE_LE.inverse_transform([pred_idx])[0]
            prob_dict = {str(VOICE_LE.classes_[i]): float(probs[i]) for i in range(len(probs))}
            return {"predicted_label": str(label), "probabilities": prob_dict}
        else:
            pred_idx = int(VOICE_CLF.predict(X)[0])
            label = VOICE_LE.inverse_transform([pred_idx])[0]
            return {"predicted_label": str(label), "probabilities": {}}
    except Exception as e:
        return {"error": "prediction_failed", "details": str(e)}

# ---------------------- Audio upload handling ----------------------
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".m4p", ".webm", ".ogg"}
_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
}

def _get_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower().strip()

def save_uploaded_audio(file_storage) -> str:
    ext = _get_extension(file_storage.filename)
    if not ext:
        ext = _MIME_TO_EXT.get((file_storage.mimetype or "").lower(), "")

    if ext not in ALLOWED_AUDIO_EXTS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTS)}. "
            f"Got filename='{file_storage.filename}', mimetype='{file_storage.mimetype}'"
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    file_storage.save(tmp_path)
    return tmp_path

# ---------------------- STT (Whisper) ----------------------
def transcribe_audio_from_path(path: str) -> str:
    # If silence -> return empty transcript (so grading becomes INCORRECT, and voice confidence is NO_SPEECH)
    if _is_silence(path):
        return ""

    try:
        result = WHISPER_MODEL.transcribe(path, language="en", fp16=False)
        text = result.get("text", "") or ""
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Local STT failed: {e}")
        return ""

# ---------------------- Attempt logging ----------------------
def log_voice_attempt(question_id, question_text, transcript, grade):
    log_path = os.path.join(BASE_DIR, "voice_attempts_log.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_id": question_id,
        "question_text": question_text,
        "transcript": transcript,
        "grade_level": grade.get("level"),
        "grade_marks": grade.get("marks"),
        "coverage": grade.get("coverage"),
    }
    try:
        df_log = pd.DataFrame([row])
        if os.path.exists(log_path):
            df_log.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df_log.to_csv(log_path, mode="w", header=True, index=False)
    except Exception as e:
        print(f"[WARN] Failed to log voice attempt: {e}")

# ---------------------- Flask app ----------------------
app = Flask(__name__)

QUESTION_CSV_PATH = os.environ.get(
    "QUESTION_CSV_PATH",
    os.path.join(BASE_DIR, "it_short_answer_dataset.csv")
)

df_questions = None

def load_question_bank():
    global df_questions
    try:
        try:
            df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="utf-8")
        except Exception:
            df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="latin-1")
        print(f"[INFO] Loaded question bank from {QUESTION_CSV_PATH} with {len(df_questions)} rows.")
    except Exception as e:
        print(f"[ERROR] Failed to load question bank CSV: {e}")
        df_questions = None

def get_question_row(question_id=None, question_text=None):
    if df_questions is None:
        return None, "Question bank not loaded."

    if question_id:
        if "question_id" not in df_questions.columns:
            return None, "question_id column not found in CSV."
        rows = df_questions[df_questions["question_id"].astype(str) == str(question_id)]
        if rows.empty:
            return None, f"question_id '{question_id}' not found."
        return rows.iloc[0], None

    if question_text:
        if "question_text" not in df_questions.columns:
            return None, "question_text column not found in CSV."
        rows = df_questions[df_questions["question_text"].astype(str) == str(question_text)]
        if rows.empty:
            return None, "question_text not found in question bank."
        return rows.iloc[0], None

    return None, "question_id or question_text required."

# Initial load
load_question_bank()
_load_voice_model()

# ---------------------- Routes ----------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "grading_mode": "ideal_answer_word_coverage_only",
            "base_dir": BASE_DIR,
            "questions_loaded": df_questions is not None,
            "num_questions": int(len(df_questions)) if df_questions is not None else 0,
            "question_csv_path": QUESTION_CSV_PATH,
            "voice_model_path": VOICE_MODEL_PATH,
            "voice_model_loaded": VOICE_CLF is not None,
            "voice_backend": VOICE_BACKEND if VOICE_CLF is not None else None,
            "voice_classes": list(VOICE_LE.classes_) if VOICE_LE is not None else [],
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD,
            "ffmpeg_available": _ffmpeg_exists(),
            "audio_allowed": sorted(list(ALLOWED_AUDIO_EXTS)),
        }
    )

@app.route("/admin/upload-questions", methods=["POST"])
def upload_questions():
    if "file" not in request.files:
        return jsonify({"error": "No file part 'file' in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        file.save(QUESTION_CSV_PATH)
        load_question_bank()
    except Exception as e:
        return jsonify({"error": f"Failed to save or load CSV: {str(e)}"}), 500

    return jsonify(
        {
            "message": "Question CSV uploaded and loaded successfully.",
            "rows": int(len(df_questions)) if df_questions is not None else 0,
        }
    )

@app.route("/questions/random", methods=["GET"])
def random_questions():
    if df_questions is None:
        return jsonify({"error": "Question bank not loaded"}), 500

    try:
        count = int(request.args.get("count", 10))
    except ValueError:
        count = 10

    sample_df = df_questions.sample(n=min(count, len(df_questions)), random_state=None)
    questions = []
    for _, row in sample_df.iterrows():
        questions.append(
            {
                "question_id": str(row.get("question_id", "")),
                "question_text": row.get("question_text", ""),
                "ideal_answer": row.get("ideal_answer", ""),
            }
        )
    return jsonify({"count": len(questions), "questions": questions})

@app.route("/grade-text", methods=["POST"])
def grade_text():
    if df_questions is None:
        return jsonify({"error": "Question bank not loaded"}), 500

    data = request.get_json(silent=True) or {}
    question_id = data.get("question_id")
    question_text = data.get("question_text")
    student_answer = data.get("student_answer", "")

    if not student_answer:
        return jsonify({"error": "student_answer is required"}), 400

    row, err = get_question_row(question_id=question_id, question_text=question_text)
    if row is None:
        return jsonify({"error": err}), 404

    ideal_answer = str(row.get("ideal_answer", "") or "").strip()
    if not ideal_answer:
        return jsonify({"error": "ideal_answer is empty in dataset for this question"}), 400

    wc = ideal_word_coverage(student_answer, ideal_answer, fuzzy_thr=0.82)
    grading = grade_by_ideal_coverage(wc["coverage"])

    return jsonify(
        {
            "question_id": str(row.get("question_id", "")),
            "question_text": row.get("question_text", ""),
            "student_answer": student_answer,
            "ideal_answer": ideal_answer,
            "ideal_word_check": wc,
            "grade": grading,
        }
    )

@app.route("/grade-voice", methods=["POST"])
def grade_voice():
    if df_questions is None:
        return jsonify({"error": "Question bank not loaded"}), 500

    question_id = request.form.get("question_id")
    question_text = request.form.get("question_text")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file part 'audio' in request"}), 400

    audio_file = request.files["audio"]

    try:
        tmp_path = save_uploaded_audio(audio_file)
        wav_path = ensure_wav(tmp_path)

        transcript = transcribe_audio_from_path(wav_path)
        voice_conf = predict_voice_confidence(wav_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        for p in {locals().get("tmp_path"), locals().get("wav_path")}:
            if not p:
                continue
            try:
                os.remove(p)
            except OSError:
                pass

    row, err = get_question_row(question_id=question_id, question_text=question_text)
    if row is None:
        return jsonify({"error": err}), 404

    ideal_answer = str(row.get("ideal_answer", "") or "").strip()
    if not ideal_answer:
        return jsonify({"error": "ideal_answer is empty in dataset for this question"}), 400

    wc = ideal_word_coverage(transcript, ideal_answer, fuzzy_thr=0.82)
    grading = grade_by_ideal_coverage(wc["coverage"])

    log_voice_attempt(
        question_id=str(row.get("question_id", "")),
        question_text=row.get("question_text", ""),
        transcript=transcript,
        grade=grading,
    )

    return jsonify(
        {
            "question_id": str(row.get("question_id", "")),
            "question_text": row.get("question_text", ""),
            "transcript": transcript,
            "ideal_answer": ideal_answer,
            "ideal_word_check": wc,
            "grade": grading,
            "voice_confidence": voice_conf,
        }
    )

@app.route("/voice-confidence", methods=["POST"])
def voice_confidence_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file part 'audio' in request"}), 400

    audio_file = request.files["audio"]

    try:
        tmp_path = save_uploaded_audio(audio_file)
        wav_path = ensure_wav(tmp_path)
        voice_conf = predict_voice_confidence(wav_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        for p in {locals().get("tmp_path"), locals().get("wav_path")}:
            if not p:
                continue
            try:
                os.remove(p)
            except OSError:
                pass

    return jsonify({"voice_confidence": voice_conf})

@app.route("/")
def index_page():
    return send_from_directory(FRONTEND_DIR, "index.html")

# Serve frontend static files (css/js/images) from Frontend folder
# IMPORTANT: keep this AFTER all API routes
@app.route("/<path:filename>")
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=14764, debug=False, use_reloader=False)
