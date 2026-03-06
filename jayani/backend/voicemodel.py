

# """
# voicemodel.py
# --------------------------------------------
# Flask ICT Short Answer + Voice API module
# FastAPI app.py එක තුලින් WSGIMiddleware හරහා mount කරලා run කරනවා.

# ✅ Uses backend/models/ for ALL files:
#    - voice_confidence_model.joblib
#    - it_short_answer_dataset.csv
#    - short_answer_grader_classifier.joblib (optional)
#    - short_answer_grader_regressor.joblib  (optional)

# ✅ Topic column handling (Topics -> topic)
# ✅ /questions/random always returns topic
# ✅ /grade-voice and /grade-text always returns topic

# ✅ FIXED KEYWORD GRADING (IMPORTANT):
#    - ideal_answer = main base
#    - keywords_main = required boost
#    - keywords_optional = bonus only (NOT penalty)
#    - final_score = 0.70*ideal_cov + 0.25*main_cov + 0.05*opt_cov
#    - marks = round(final_score * 10)

# ✅ ML grading (optional):
#    combined = topic [SEP] question [SEP] answer
# """

# import os
# import re
# import tempfile
# from datetime import datetime
# import subprocess
# import difflib

# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# import librosa
# import joblib

# # ---------------------- Base dir ----------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../backend
# MODELS_DIR = os.path.join(BASE_DIR, "models")           # .../backend/models

# voice_app = Flask(__name__)

# # ---------------------- Whisper (Local STT) -----------------------
# import whisper
# print("[WHISPER] Loading local Whisper model 'base' ...")
# WHISPER_MODEL = whisper.load_model("base")
# print("[WHISPER] Model loaded.")

# # ---------------------- Text Normalization ------------------------
# STOPWORDS = {
#     "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by",
#     "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
#     "those", "it", "its", "as", "at", "from", "into", "about", "than", "then",
# }

# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def tokenize_important(text: str) -> list:
#     text = clean_text(text)
#     toks = [t for t in text.split() if len(t) >= 2 and t not in STOPWORDS]
#     seen = set()
#     out = []
#     for t in toks:
#         if t not in seen:
#             out.append(t)
#             seen.add(t)
#     return out

# def fuzzy_token_match(token: str, transcript_tokens: list, thr: float = 0.82):
#     if token in transcript_tokens:
#         return True, token

#     best = ("", 0.0)
#     for w in transcript_tokens:
#         if abs(len(w) - len(token)) >= 5:
#             continue
#         r = difflib.SequenceMatcher(None, token, w).ratio()
#         if r > best[1]:
#             best = (w, r)

#     if best[1] >= thr:
#         return True, best[0]
#     return False, ""

# def ideal_word_coverage(transcript: str, ideal_answer: str, fuzzy_thr: float = 0.82) -> dict:
#     ideal_tokens = tokenize_important(ideal_answer)
#     trans_tokens = tokenize_important(transcript)

#     matched = []
#     unmatched = []
#     matches_map = {}

#     if len(ideal_tokens) == 0:
#         return {
#             "coverage": 0.0,
#             "ideal_tokens": [],
#             "transcript_tokens": trans_tokens,
#             "matched_tokens": [],
#             "unmatched_tokens": [],
#             "matches_map": {},
#             "note": "ideal_answer_has_no_tokens_after_cleaning",
#         }

#     for t in ideal_tokens:
#         ok, m = fuzzy_token_match(t, trans_tokens, thr=fuzzy_thr)
#         if ok:
#             matched.append(t)
#             matches_map[t] = m
#         else:
#             unmatched.append(t)

#     coverage = len(matched) / len(ideal_tokens)
#     return {
#         "coverage": float(coverage),
#         "ideal_tokens": ideal_tokens,
#         "transcript_tokens": trans_tokens,
#         "matched_tokens": matched,
#         "unmatched_tokens": unmatched,
#         "matches_map": matches_map,
#         "fuzzy_threshold": float(fuzzy_thr),
#     }

# # ---------------------- ✅ FIXED: keyword parsing + weighted grading ----------------------
# def _split_keywords(s: str):
#     """supports: 'a|b|c' and 'a,b,c' """
#     if not s:
#         return []
#     s = str(s).replace(",", "|")
#     return [p.strip() for p in s.split("|") if p.strip()]

# def _keywords_text_from_row(row):
#     km = str(row.get("keywords_main", "") or row.get("Keywords_main", "") or "").strip()
#     ko = str(row.get("keywords_optional", "") or row.get("Keywords_optional", "") or "").strip()

#     main_list = _split_keywords(km)
#     opt_list  = _split_keywords(ko)

#     main_text = " ".join(main_list).strip()
#     opt_text  = " ".join(opt_list).strip()
#     return main_text, opt_text, main_list, opt_list

# def grade_rule_based_weighted(student_text: str, row, fuzzy_thr: float = 0.82):
#     """
#     final_score = 0.70*ideal_cov + 0.25*main_cov + 0.05*opt_cov
#     marks = round(final_score * 10)
#     """
#     ideal = str(row.get("ideal_answer", "") or "").strip()
#     main_text, opt_text, main_list, opt_list = _keywords_text_from_row(row)

#     if not ideal and not main_text and not opt_text:
#         return None, {"error": "ideal_and_keywords_empty"}

#     ideal_cov = ideal_word_coverage(student_text, ideal, fuzzy_thr=fuzzy_thr)["coverage"] if ideal else 0.0
#     main_cov  = ideal_word_coverage(student_text, main_text, fuzzy_thr=fuzzy_thr)["coverage"] if main_text else 0.0
#     opt_cov   = ideal_word_coverage(student_text, opt_text, fuzzy_thr=fuzzy_thr)["coverage"] if opt_text else 0.0

#     final_score = (0.70 * ideal_cov) + (0.25 * main_cov) + (0.05 * opt_cov)
#     final_score = max(0.0, min(1.0, float(final_score)))

#     marks = int(round(final_score * 10))

#     if marks >= 8:
#         level = "GOOD"
#     elif marks >= 5:
#         level = "PARTIAL"
#     elif marks >= 3:
#         level = "WEAK"
#     else:
#         level = "INCORRECT"

#     grading = {"level": level, "marks": marks, "coverage": float(final_score)}
#     debug = {
#         "ideal_cov": float(ideal_cov),
#         "keywords_main_cov": float(main_cov),
#         "keywords_optional_cov": float(opt_cov),
#         "final_score": float(final_score),
#         "main_keywords": main_list,
#         "optional_keywords": opt_list,
#         "fuzzy_threshold": float(fuzzy_thr),
#     }
#     return grading, debug

# # ---------------------- Voice confidence model ---------------------
# VOICE_MODEL_PATH = os.environ.get(
#     "VOICE_MODEL_PATH",
#     os.path.join(MODELS_DIR, "voice_confidence_model.joblib")
# )

# VOICE_CLF = None
# VOICE_LE = None
# VOICE_SAMPLE_RATE = 16000

# VOICE_BACKEND = "mfcc"  # "mfcc" or "yamnet"
# YAMNET_HANDLE = None
# EMBED_POOL = "mean_std"
# _YAMNET_MODEL = None

# def _load_voice_model():
#     global VOICE_CLF, VOICE_LE, VOICE_SAMPLE_RATE, VOICE_BACKEND, YAMNET_HANDLE, EMBED_POOL
#     try:
#         bundle = joblib.load(VOICE_MODEL_PATH)
#         VOICE_CLF = bundle["model"]
#         VOICE_LE = bundle["label_encoder"]
#         VOICE_SAMPLE_RATE = bundle.get("sample_rate", VOICE_SAMPLE_RATE)

#         VOICE_BACKEND = bundle.get("embedding_backend", "mfcc")
#         YAMNET_HANDLE = bundle.get("yamnet_handle", None)
#         EMBED_POOL = bundle.get("embed_pool", EMBED_POOL)

#         print("[VOICE] Loaded model from:", VOICE_MODEL_PATH)
#         print("[VOICE] Backend:", VOICE_BACKEND)
#         print("[VOICE] Classes:", list(VOICE_LE.classes_))

#         if VOICE_BACKEND == "yamnet" and not YAMNET_HANDLE:
#             raise RuntimeError("backend=yamnet but 'yamnet_handle' missing in model bundle.")
#     except Exception as e:
#         print(f"[VOICE] Could not load voice confidence model: {e}")
#         VOICE_CLF = None
#         VOICE_LE = None

# def _get_yamnet():
#     global _YAMNET_MODEL
#     if _YAMNET_MODEL is None:
#         import tensorflow_hub as hub
#         print(f"[VOICE] Loading YAMNet from TF Hub: {YAMNET_HANDLE}")
#         _YAMNET_MODEL = hub.load(YAMNET_HANDLE)
#     return _YAMNET_MODEL

# def _extract_yamnet_embedding(path: str):
#     try:
#         import tensorflow as tf
#     except Exception as e:
#         print("[VOICE] TensorFlow not available:", e)
#         return None

#     try:
#         y, _sr = librosa.load(path, sr=VOICE_SAMPLE_RATE, mono=True)
#     except Exception as e:
#         print(f"[VOICE] Failed to load audio '{path}': {e}")
#         return None
#     if y.size == 0:
#         return None

#     waveform = tf.convert_to_tensor(y, dtype=tf.float32)
#     yamnet = _get_yamnet()
#     _scores, embeddings, _spectrogram = yamnet(waveform)

#     emb = embeddings.numpy()
#     if emb.size == 0:
#         return None

#     if EMBED_POOL == "mean":
#         vec = emb.mean(axis=0)
#     elif EMBED_POOL == "mean_std":
#         vec = np.concatenate([emb.mean(axis=0), emb.std(axis=0)], axis=0)
#     else:
#         return None

#     return vec.astype(np.float32)

# def _extract_mfcc_features(path: str):
#     try:
#         y, sr = librosa.load(path, sr=VOICE_SAMPLE_RATE, mono=True)
#     except Exception as e:
#         print(f"[VOICE] Failed to load audio '{path}': {e}")
#         return None
#     if y.size == 0:
#         return None

#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = mfcc.mean(axis=1)
#     mfcc_std = mfcc.std(axis=1)

#     zcr = librosa.feature.zero_crossing_rate(y)[0]
#     zcr_mean = float(zcr.mean())
#     zcr_std = float(zcr.std())

#     rms = librosa.feature.rms(y=y)[0]
#     rms_mean = float(rms.mean())
#     rms_std = float(rms.std())

#     try:
#         tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#         tempo = float(np.array(tempo).ravel()[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
#     except Exception:
#         tempo = 0.0

#     stats_vec = np.array([zcr_mean, zcr_std, rms_mean, rms_std, tempo], dtype=np.float32)
#     feats = np.concatenate([mfcc_mean, mfcc_std, stats_vec]).astype(np.float32)
#     return feats

# # ---------------------- SILENCE / NO SPEECH ----------------------
# SILENCE_RMS_THRESHOLD = float(os.environ.get("SILENCE_RMS_THRESHOLD", "0.008"))

# def _rms_mean(path: str, sr: int = 16000) -> float:
#     try:
#         y, _ = librosa.load(path, sr=sr, mono=True)
#         if y.size == 0:
#             return 0.0
#         rms = librosa.feature.rms(y=y)[0]
#         return float(rms.mean()) if rms.size else 0.0
#     except Exception:
#         return 0.0

# def _is_silence(path: str) -> bool:
#     return _rms_mean(path, sr=VOICE_SAMPLE_RATE) < SILENCE_RMS_THRESHOLD

# # ---------------------- FFmpeg convert to wav ----------------------
# def _ffmpeg_exists() -> bool:
#     try:
#         subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         return True
#     except Exception:
#         return False

# def ensure_wav(path: str) -> str:
#     ext = os.path.splitext(path)[1].lower()
#     if ext == ".wav":
#         return path
#     if not _ffmpeg_exists():
#         return path  # fallback

#     out_wav = path + ".wav"
#     try:
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(VOICE_SAMPLE_RATE), out_wav],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             check=True
#         )
#         return out_wav
#     except Exception as e:
#         print("[AUDIO] ffmpeg convert failed:", e)
#         return path

# def predict_voice_confidence(path: str) -> dict:
#     if VOICE_CLF is None or VOICE_LE is None:
#         return {"error": "voice_model_not_loaded"}

#     if _is_silence(path):
#         return {
#             "predicted_label": "NO_SPEECH",
#             "probabilities": {},
#             "note": f"silence_detected_rms<thr({SILENCE_RMS_THRESHOLD})"
#         }

#     vec = _extract_yamnet_embedding(path) if VOICE_BACKEND == "yamnet" else _extract_mfcc_features(path)
#     if vec is None:
#         return {"error": "could_not_extract_features"}

#     X = vec.reshape(1, -1)

#     try:
#         if hasattr(VOICE_CLF, "predict_proba"):
#             probs = VOICE_CLF.predict_proba(X)[0]
#             pred_idx = int(np.argmax(probs))
#             label = VOICE_LE.inverse_transform([pred_idx])[0]
#             prob_dict = {str(VOICE_LE.classes_[i]): float(probs[i]) for i in range(len(probs))}
#             return {"predicted_label": str(label), "probabilities": prob_dict}
#         else:
#             pred_idx = int(VOICE_CLF.predict(X)[0])
#             label = VOICE_LE.inverse_transform([pred_idx])[0]
#             return {"predicted_label": str(label), "probabilities": {}}
#     except Exception as e:
#         return {"error": "prediction_failed", "details": str(e)}

# # ---------------------- Audio upload handling ----------------------
# ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".m4p", ".webm", ".ogg"}
# _MIME_TO_EXT = {
#     "audio/wav": ".wav",
#     "audio/x-wav": ".wav",
#     "audio/mpeg": ".mp3",
#     "audio/mp3": ".mp3",
#     "audio/mp4": ".m4a",
#     "audio/aac": ".m4a",
#     "audio/webm": ".webm",
#     "audio/ogg": ".ogg",
#     "application/ogg": ".ogg",
# }

# def _get_extension(filename: str) -> str:
#     return os.path.splitext(filename or "")[1].lower().strip()

# def save_uploaded_audio(file_storage) -> str:
#     ext = _get_extension(file_storage.filename)
#     if not ext:
#         ext = _MIME_TO_EXT.get((file_storage.mimetype or "").lower(), "")

#     if ext not in ALLOWED_AUDIO_EXTS:
#         raise ValueError(
#             f"Unsupported audio format '{ext}'. Allowed: {sorted(ALLOWED_AUDIO_EXTS)}. "
#             f"Got filename='{file_storage.filename}', mimetype='{file_storage.mimetype}'"
#         )

#     tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
#     os.close(tmp_fd)
#     file_storage.save(tmp_path)
#     return tmp_path

# # ---------------------- STT (Whisper) ----------------------
# def transcribe_audio_from_path(path: str) -> str:
#     if _is_silence(path):
#         return ""
#     try:
#         result = WHISPER_MODEL.transcribe(path, language="en", fp16=False)
#         text = result.get("text", "") or ""
#         return clean_text(text)
#     except Exception as e:
#         print(f"[ERROR] Local STT failed: {e}")
#         return ""

# # ---------------------- Attempt logging ----------------------
# def log_voice_attempt(question_id, topic, question_text, transcript, grade):
#     log_path = os.path.join(BASE_DIR, "voice_attempts_log.csv")
#     row = {
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#         "question_id": question_id,
#         "topic": topic,
#         "question_text": question_text,
#         "transcript": transcript,
#         "grade_level": grade.get("level"),
#         "grade_marks": grade.get("marks"),
#         "coverage": grade.get("coverage"),
#     }
#     try:
#         df_log = pd.DataFrame([row])
#         if os.path.exists(log_path):
#             df_log.to_csv(log_path, mode="a", header=False, index=False)
#         else:
#             df_log.to_csv(log_path, mode="w", header=True, index=False)
#     except Exception as e:
#         print(f"[WARN] Failed to log voice attempt: {e}")

# # ---------------------- Question Bank (models folder) ----------------------
# QUESTION_CSV_PATH = os.environ.get(
#     "QUESTION_CSV_PATH",
#     os.path.join(MODELS_DIR, "it_short_answer_dataset.csv")
# )

# df_questions = None

# def load_question_bank():
#     """
#     ✅ Supports 'Topics' column
#     - normalizes columns
#     - creates 'topic' always
#     """
#     global df_questions
#     try:
#         try:
#             df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="utf-8")
#         except Exception:
#             df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="latin-1")

#         df_questions.columns = [c.strip() for c in df_questions.columns]

#         # map Topics -> topic
#         if "topic" not in df_questions.columns and "Topics" in df_questions.columns:
#             df_questions["topic"] = df_questions["Topics"]

#         if "topic" not in df_questions.columns:
#             df_questions["topic"] = ""

#         df_questions["topic"] = df_questions["topic"].fillna("").astype(str)

#         print(f"[INFO] Loaded question bank from {QUESTION_CSV_PATH} with {len(df_questions)} rows.")
#         print("[INFO] Columns:", list(df_questions.columns))
#     except Exception as e:
#         print(f"[ERROR] Failed to load question bank CSV: {e}")
#         df_questions = None

# def get_question_row(question_id=None, question_text=None):
#     if df_questions is None:
#         return None, "Question bank not loaded."

#     if question_id:
#         if "question_id" not in df_questions.columns:
#             return None, "question_id column not found in CSV."
#         rows = df_questions[df_questions["question_id"].astype(str) == str(question_id)]
#         if rows.empty:
#             return None, f"question_id '{question_id}' not found."
#         return rows.iloc[0], None

#     if question_text:
#         if "question_text" not in df_questions.columns:
#             return None, "question_text column not found in CSV."
#         rows = df_questions[df_questions["question_text"].astype(str) == str(question_text)]
#         if rows.empty:
#             return None, "question_text not found in question bank."
#         return rows.iloc[0], None

#     return None, "question_id or question_text required."

# # ---------------------- ML Grader (optional, models folder) ----------------------
# ML_CLF_PATH = os.environ.get(
#     "ML_CLF_PATH",
#     os.path.join(MODELS_DIR, "short_answer_grader_classifier.joblib")
# )
# ML_REG_PATH = os.environ.get(
#     "ML_REG_PATH",
#     os.path.join(MODELS_DIR, "short_answer_grader_regressor.joblib")
# )

# ML_CLF = None
# ML_REG = None

# def _load_ml_grader():
#     global ML_CLF, ML_REG
#     try:
#         if os.path.exists(ML_CLF_PATH):
#             ML_CLF = joblib.load(ML_CLF_PATH)
#             print("[ML] Loaded classifier:", ML_CLF_PATH)
#         else:
#             print("[ML] Classifier not found:", ML_CLF_PATH)
#             ML_CLF = None
#     except Exception as e:
#         print("[ML] Failed to load classifier:", e)
#         ML_CLF = None

#     try:
#         if os.path.exists(ML_REG_PATH):
#             ML_REG = joblib.load(ML_REG_PATH)
#             print("[ML] Loaded regressor:", ML_REG_PATH)
#         else:
#             print("[ML] Regressor not found:", ML_REG_PATH)
#             ML_REG = None
#     except Exception as e:
#         print("[ML] Failed to load regressor:", e)
#         ML_REG = None

# def ml_grade(topic: str, question_text: str, student_answer: str) -> dict:
#     if ML_CLF is None:
#         return {"error": "ml_classifier_not_loaded"}

#     topic = clean_text(topic or "")
#     qt = clean_text(question_text or "")
#     ans = clean_text(student_answer or "")
#     combined = f"{topic} [SEP] {qt} [SEP] {ans}".strip()

#     pred_label = ML_CLF.predict([combined])[0]

#     prob_dict = {}
#     if hasattr(ML_CLF, "predict_proba"):
#         probs = ML_CLF.predict_proba([combined])[0]
#         classes = list(ML_CLF.classes_)
#         prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

#     marks_pred = None
#     if ML_REG is not None:
#         try:
#             marks_pred = float(ML_REG.predict([combined])[0])
#             marks_pred = max(0.0, min(10.0, marks_pred))
#         except Exception:
#             marks_pred = None

#     return {"predicted_label": str(pred_label), "probabilities": prob_dict, "marks_pred": marks_pred}

# # ---------------------- Initial load ----------------------
# load_question_bank()
# _load_voice_model()
# _load_ml_grader()

# # ---------------------- Routes ----------------------
# @voice_app.route("/health", methods=["GET"])
# def health():
#     return jsonify(
#         {
#             "status": "ok",
#             "base_dir": BASE_DIR,
#             "models_dir": MODELS_DIR,

#             "question_csv_path": QUESTION_CSV_PATH,
#             "questions_loaded": df_questions is not None,
#             "num_questions": int(len(df_questions)) if df_questions is not None else 0,

#             "ffmpeg_available": _ffmpeg_exists(),
#             "audio_allowed": sorted(list(ALLOWED_AUDIO_EXTS)),
#             "silence_rms_threshold": SILENCE_RMS_THRESHOLD,

#             "voice_model_path": VOICE_MODEL_PATH,
#             "voice_model_loaded": VOICE_CLF is not None,
#             "voice_backend": VOICE_BACKEND if VOICE_CLF is not None else None,
#             "voice_classes": list(VOICE_LE.classes_) if VOICE_LE is not None else [],

#             "ml_classifier_path": ML_CLF_PATH,
#             "ml_regressor_path": ML_REG_PATH,
#             "ml_classifier_loaded": ML_CLF is not None,
#             "ml_regressor_loaded": ML_REG is not None,
#         }
#     )

# @voice_app.route("/questions/random", methods=["GET"])
# def random_questions():
#     if df_questions is None:
#         return jsonify({"error": "Question bank not loaded"}), 500

#     try:
#         count = int(request.args.get("count", 10))
#     except ValueError:
#         count = 10

#     sample_df = df_questions.sample(n=min(count, len(df_questions)), random_state=None)
#     questions = []
#     for _, row in sample_df.iterrows():
#         topic = str(row.get("topic", "") or row.get("Topics", "") or "")
#         questions.append(
#             {
#                 "question_id": str(row.get("question_id", "")),
#                 "topic": topic,
#                 "question_text": row.get("question_text", ""),
#                 "ideal_answer": row.get("ideal_answer", ""),
#             }
#         )
#     return jsonify({"count": len(questions), "questions": questions})

# @voice_app.route("/grade-text", methods=["POST"])
# def grade_text():
#     if df_questions is None:
#         return jsonify({"error": "Question bank not loaded"}), 500

#     data = request.get_json(silent=True) or {}
#     question_id = data.get("question_id")
#     question_text = data.get("question_text")
#     student_answer = data.get("student_answer", "")

#     if not student_answer:
#         return jsonify({"error": "student_answer is required"}), 400

#     row, err = get_question_row(question_id=question_id, question_text=question_text)
#     if row is None:
#         return jsonify({"error": err}), 404

#     topic = str(row.get("topic", "") or row.get("Topics", "") or "")
#     ideal_answer = str(row.get("ideal_answer", "") or "").strip()

#     grading, debug = grade_rule_based_weighted(student_answer, row, fuzzy_thr=0.82)
#     if grading is None:
#         return jsonify({"error": debug.get("error", "grading_failed")}), 400

#     return jsonify(
#         {
#             "question_id": str(row.get("question_id", "")),
#             "topic": topic,
#             "question_text": row.get("question_text", ""),
#             "student_answer": student_answer,
#             "ideal_answer": ideal_answer,
#             "grade": grading,
#             "debug": debug,
#         }
#     )

# @voice_app.route("/grade-voice", methods=["POST"])
# def grade_voice():
#     if df_questions is None:
#         return jsonify({"error": "Question bank not loaded"}), 500

#     question_id = request.form.get("question_id")
#     question_text = request.form.get("question_text")

#     if "audio" not in request.files:
#         return jsonify({"error": "No audio file part 'audio' in request"}), 400

#     audio_file = request.files["audio"]

#     tmp_path = None
#     wav_path = None
#     try:
#         tmp_path = save_uploaded_audio(audio_file)
#         wav_path = ensure_wav(tmp_path)

#         transcript = transcribe_audio_from_path(wav_path)
#         voice_conf = predict_voice_confidence(wav_path)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400
#     finally:
#         for p in {tmp_path, wav_path}:
#             if not p:
#                 continue
#             try:
#                 os.remove(p)
#             except OSError:
#                 pass

#     row, err = get_question_row(question_id=question_id, question_text=question_text)
#     if row is None:
#         return jsonify({"error": err}), 404

#     topic = str(row.get("topic", "") or row.get("Topics", "") or "")
#     ideal_answer = str(row.get("ideal_answer", "") or "").strip()

#     grading, debug = grade_rule_based_weighted(transcript, row, fuzzy_thr=0.82)
#     if grading is None:
#         return jsonify({"error": debug.get("error", "grading_failed")}), 400

#     log_voice_attempt(
#         question_id=str(row.get("question_id", "")),
#         topic=topic,
#         question_text=row.get("question_text", ""),
#         transcript=transcript,
#         grade=grading,
#     )

#     return jsonify(
#         {
#             "question_id": str(row.get("question_id", "")),
#             "topic": topic,
#             "question_text": row.get("question_text", ""),
#             "transcript": transcript,
#             "ideal_answer": ideal_answer,
#             "grade": grading,
#             "debug": debug,
#             "voice_confidence": voice_conf,
#         }
#     )

# @voice_app.route("/grade-text-ml", methods=["POST"])
# def grade_text_ml():
#     data = request.get_json(silent=True) or {}
#     topic = data.get("topic", "")
#     question_text = data.get("question_text", "")
#     student_answer = data.get("student_answer", "")

#     if not question_text or not student_answer:
#         return jsonify({"error": "question_text and student_answer required"}), 400

#     out = ml_grade(topic, question_text, student_answer)
#     return jsonify({"topic": topic, "question_text": question_text, "student_answer": student_answer, "ml": out})

# @voice_app.route("/voice-confidence", methods=["POST"])
# def voice_confidence_route():
#     if "audio" not in request.files:
#         return jsonify({"error": "No audio file part 'audio' in request"}), 400

#     audio_file = request.files["audio"]

#     tmp_path = None
#     wav_path = None
#     try:
#         tmp_path = save_uploaded_audio(audio_file)
#         wav_path = ensure_wav(tmp_path)
#         voice_conf = predict_voice_confidence(wav_path)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400
#     finally:
#         for p in {tmp_path, wav_path}:
#             if not p:
#                 continue
#             try:
#                 os.remove(p)
#             except OSError:
#                 pass

#     return jsonify({"voice_confidence": voice_conf})

"""
voicemodel.py
--------------------------------------------
Flask ICT Short Answer + Voice API module
FastAPI app.py එක තුලින් WSGIMiddleware හරහා mount කරලා run කරනවා.

✅ Uses backend/models/ for ALL files:
   - voice_confidence_model.joblib
   - it_short_answer_dataset.csv
   - short_answer_grader_classifier.joblib (optional)
   - short_answer_grader_regressor.joblib  (optional)

✅ Topic column handling (Topics -> topic)
✅ /questions/random always returns topic
✅ /grade-voice and /grade-text always returns topic

✅ FIXED KEYWORD GRADING (IMPORTANT):
   - ideal_answer = main base
   - keywords_main = required boost
   - keywords_optional = bonus only (NOT penalty)
   - final_score = 0.70*ideal_cov + 0.25*main_cov + 0.05*opt_cov
   - marks = round(final_score * 10)

✅ ML grading (optional):
   combined = topic [SEP] question [SEP] answer
"""

import os
import re
import tempfile
from datetime import datetime
import subprocess
import difflib

# ---------------------- Base dir + cache ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../backend
MODELS_DIR = os.path.join(BASE_DIR, "models")           # .../backend/models
TFHUB_CACHE_DIR = os.path.join(BASE_DIR, "tfhub_cache")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TFHUB_CACHE_DIR, exist_ok=True)

# Keep TF Hub files inside project, not temp
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import librosa
import joblib

voice_app = Flask(__name__)

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


# ---------------------- keyword parsing + weighted grading ----------------------
def _split_keywords(s: str):
    """supports: 'a|b|c' and 'a,b,c'"""
    if not s:
        return []
    s = str(s).replace(",", "|")
    return [p.strip() for p in s.split("|") if p.strip()]


def _keywords_text_from_row(row):
    km = str(row.get("keywords_main", "") or row.get("Keywords_main", "") or "").strip()
    ko = str(row.get("keywords_optional", "") or row.get("Keywords_optional", "") or "").strip()

    main_list = _split_keywords(km)
    opt_list = _split_keywords(ko)

    main_text = " ".join(main_list).strip()
    opt_text = " ".join(opt_list).strip()
    return main_text, opt_text, main_list, opt_list


def grade_rule_based_weighted(student_text: str, row, fuzzy_thr: float = 0.82):
    """
    final_score = 0.70*ideal_cov + 0.25*main_cov + 0.05*opt_cov
    marks = round(final_score * 10)
    """
    ideal = str(row.get("ideal_answer", "") or "").strip()
    main_text, opt_text, main_list, opt_list = _keywords_text_from_row(row)

    if not ideal and not main_text and not opt_text:
        return None, {"error": "ideal_and_keywords_empty"}

    ideal_cov = ideal_word_coverage(student_text, ideal, fuzzy_thr=fuzzy_thr)["coverage"] if ideal else 0.0
    main_cov = ideal_word_coverage(student_text, main_text, fuzzy_thr=fuzzy_thr)["coverage"] if main_text else 0.0
    opt_cov = ideal_word_coverage(student_text, opt_text, fuzzy_thr=fuzzy_thr)["coverage"] if opt_text else 0.0

    final_score = (0.70 * ideal_cov) + (0.25 * main_cov) + (0.05 * opt_cov)
    final_score = max(0.0, min(1.0, float(final_score)))

    marks = int(round(final_score * 10))

    if marks >= 8:
        level = "GOOD"
    elif marks >= 5:
        level = "PARTIAL"
    elif marks >= 3:
        level = "WEAK"
    else:
        level = "INCORRECT"

    grading = {"level": level, "marks": marks, "coverage": float(final_score)}
    debug = {
        "ideal_cov": float(ideal_cov),
        "keywords_main_cov": float(main_cov),
        "keywords_optional_cov": float(opt_cov),
        "final_score": float(final_score),
        "main_keywords": main_list,
        "optional_keywords": opt_list,
        "fuzzy_threshold": float(fuzzy_thr),
    }
    return grading, debug


# ---------------------- Voice confidence model ---------------------
VOICE_MODEL_PATH = os.environ.get(
    "VOICE_MODEL_PATH",
    os.path.join(MODELS_DIR, "voice_confidence_model.joblib")
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
        print(f"[VOICE] TFHUB_CACHE_DIR = {os.environ.get('TFHUB_CACHE_DIR')}")
        print(f"[VOICE] Loading YAMNet from TF Hub: {YAMNET_HANDLE}")
        _YAMNET_MODEL = hub.load(YAMNET_HANDLE)
        print("[VOICE] YAMNet loaded successfully.")
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


# ---------------------- SILENCE / NO SPEECH ----------------------
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


# ---------------------- FFmpeg convert to wav ----------------------
def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def ensure_wav(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path

    if not _ffmpeg_exists():
        return path

    out_wav = path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(VOICE_SAMPLE_RATE), out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return out_wav
    except Exception as e:
        print("[AUDIO] ffmpeg convert failed:", e)
        return path


def predict_voice_confidence(path: str) -> dict:
    if VOICE_CLF is None or VOICE_LE is None:
        return {"error": "voice_model_not_loaded"}

    if _is_silence(path):
        return {
            "predicted_label": "NO_SPEECH",
            "probabilities": {},
            "note": f"silence_detected_rms<thr({SILENCE_RMS_THRESHOLD})",
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
def log_voice_attempt(question_id, topic, question_text, transcript, grade):
    log_path = os.path.join(BASE_DIR, "voice_attempts_log.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_id": question_id,
        "topic": topic,
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


# ---------------------- Question Bank ----------------------
QUESTION_CSV_PATH = os.environ.get(
    "QUESTION_CSV_PATH",
    os.path.join(MODELS_DIR, "it_short_answer_dataset.csv")
)

df_questions = None


def load_question_bank():
    global df_questions
    try:
        try:
            df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="utf-8")
        except Exception:
            df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding="latin-1")

        df_questions.columns = [c.strip() for c in df_questions.columns]

        if "topic" not in df_questions.columns and "Topics" in df_questions.columns:
            df_questions["topic"] = df_questions["Topics"]

        if "topic" not in df_questions.columns:
            df_questions["topic"] = ""

        df_questions["topic"] = df_questions["topic"].fillna("").astype(str)

        print(f"[INFO] Loaded question bank from {QUESTION_CSV_PATH} with {len(df_questions)} rows.")
        print("[INFO] Columns:", list(df_questions.columns))
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


# ---------------------- ML Grader ----------------------
ML_CLF_PATH = os.environ.get(
    "ML_CLF_PATH",
    os.path.join(MODELS_DIR, "short_answer_grader_classifier.joblib")
)
ML_REG_PATH = os.environ.get(
    "ML_REG_PATH",
    os.path.join(MODELS_DIR, "short_answer_grader_regressor.joblib")
)

ML_CLF = None
ML_REG = None


def _load_ml_grader():
    global ML_CLF, ML_REG

    try:
        if os.path.exists(ML_CLF_PATH):
            ML_CLF = joblib.load(ML_CLF_PATH)
            print("[ML] Loaded classifier:", ML_CLF_PATH)
        else:
            print("[ML] Classifier not found:", ML_CLF_PATH)
            ML_CLF = None
    except Exception as e:
        print("[ML] Failed to load classifier:", e)
        ML_CLF = None

    try:
        if os.path.exists(ML_REG_PATH):
            ML_REG = joblib.load(ML_REG_PATH)
            print("[ML] Loaded regressor:", ML_REG_PATH)
        else:
            print("[ML] Regressor not found:", ML_REG_PATH)
            ML_REG = None
    except Exception as e:
        print("[ML] Failed to load regressor:", e)
        ML_REG = None


def ml_grade(topic: str, question_text: str, student_answer: str) -> dict:
    if ML_CLF is None:
        return {"error": "ml_classifier_not_loaded"}

    topic = clean_text(topic or "")
    qt = clean_text(question_text or "")
    ans = clean_text(student_answer or "")
    combined = f"{topic} [SEP] {qt} [SEP] {ans}".strip()

    pred_label = ML_CLF.predict([combined])[0]

    prob_dict = {}
    if hasattr(ML_CLF, "predict_proba"):
        probs = ML_CLF.predict_proba([combined])[0]
        classes = list(ML_CLF.classes_)
        prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

    marks_pred = None
    if ML_REG is not None:
        try:
            marks_pred = float(ML_REG.predict([combined])[0])
            marks_pred = max(0.0, min(10.0, marks_pred))
        except Exception:
            marks_pred = None

    return {
        "predicted_label": str(pred_label),
        "probabilities": prob_dict,
        "marks_pred": marks_pred,
    }


# ---------------------- Initial load ----------------------
load_question_bank()
_load_voice_model()
_load_ml_grader()

# ---------------------- Routes ----------------------
@voice_app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "base_dir": BASE_DIR,
            "models_dir": MODELS_DIR,
            "tfhub_cache_dir": os.environ.get("TFHUB_CACHE_DIR"),
            "question_csv_path": QUESTION_CSV_PATH,
            "questions_loaded": df_questions is not None,
            "num_questions": int(len(df_questions)) if df_questions is not None else 0,
            "ffmpeg_available": _ffmpeg_exists(),
            "audio_allowed": sorted(list(ALLOWED_AUDIO_EXTS)),
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD,
            "voice_model_path": VOICE_MODEL_PATH,
            "voice_model_loaded": VOICE_CLF is not None,
            "voice_backend": VOICE_BACKEND if VOICE_CLF is not None else None,
            "voice_classes": list(VOICE_LE.classes_) if VOICE_LE is not None else [],
            "ml_classifier_path": ML_CLF_PATH,
            "ml_regressor_path": ML_REG_PATH,
            "ml_classifier_loaded": ML_CLF is not None,
            "ml_regressor_loaded": ML_REG is not None,
        }
    )


@voice_app.route("/questions/random", methods=["GET"])
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
        topic = str(row.get("topic", "") or row.get("Topics", "") or "")
        questions.append(
            {
                "question_id": str(row.get("question_id", "")),
                "topic": topic,
                "question_text": row.get("question_text", ""),
                "ideal_answer": row.get("ideal_answer", ""),
            }
        )
    return jsonify({"count": len(questions), "questions": questions})


@voice_app.route("/grade-text", methods=["POST"])
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

    topic = str(row.get("topic", "") or row.get("Topics", "") or "")
    ideal_answer = str(row.get("ideal_answer", "") or "").strip()

    grading, debug = grade_rule_based_weighted(student_answer, row, fuzzy_thr=0.82)
    if grading is None:
        return jsonify({"error": debug.get("error", "grading_failed")}), 400

    return jsonify(
        {
            "question_id": str(row.get("question_id", "")),
            "topic": topic,
            "question_text": row.get("question_text", ""),
            "student_answer": student_answer,
            "ideal_answer": ideal_answer,
            "grade": grading,
            "debug": debug,
        }
    )


@voice_app.route("/grade-voice", methods=["POST"])
def grade_voice():
    if df_questions is None:
        return jsonify({"error": "Question bank not loaded"}), 500

    question_id = request.form.get("question_id")
    question_text = request.form.get("question_text")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file part 'audio' in request"}), 400

    audio_file = request.files["audio"]

    tmp_path = None
    wav_path = None
    try:
        tmp_path = save_uploaded_audio(audio_file)
        wav_path = ensure_wav(tmp_path)

        transcript = transcribe_audio_from_path(wav_path)
        voice_conf = predict_voice_confidence(wav_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        for p in {tmp_path, wav_path}:
            if not p:
                continue
            try:
                os.remove(p)
            except OSError:
                pass

    row, err = get_question_row(question_id=question_id, question_text=question_text)
    if row is None:
        return jsonify({"error": err}), 404

    topic = str(row.get("topic", "") or row.get("Topics", "") or "")
    ideal_answer = str(row.get("ideal_answer", "") or "").strip()

    grading, debug = grade_rule_based_weighted(transcript, row, fuzzy_thr=0.82)
    if grading is None:
        return jsonify({"error": debug.get("error", "grading_failed")}), 400

    log_voice_attempt(
        question_id=str(row.get("question_id", "")),
        topic=topic,
        question_text=row.get("question_text", ""),
        transcript=transcript,
        grade=grading,
    )

    return jsonify(
        {
            "question_id": str(row.get("question_id", "")),
            "topic": topic,
            "question_text": row.get("question_text", ""),
            "transcript": transcript,
            "ideal_answer": ideal_answer,
            "grade": grading,
            "debug": debug,
            "voice_confidence": voice_conf,
        }
    )


@voice_app.route("/grade-text-ml", methods=["POST"])
def grade_text_ml():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    question_text = data.get("question_text", "")
    student_answer = data.get("student_answer", "")

    if not question_text or not student_answer:
        return jsonify({"error": "question_text and student_answer required"}), 400

    out = ml_grade(topic, question_text, student_answer)
    return jsonify(
        {
            "topic": topic,
            "question_text": question_text,
            "student_answer": student_answer,
            "ml": out,
        }
    )


@voice_app.route("/voice-confidence", methods=["POST"])
def voice_confidence_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file part 'audio' in request"}), 400

    audio_file = request.files["audio"]

    tmp_path = None
    wav_path = None
    try:
        tmp_path = save_uploaded_audio(audio_file)
        wav_path = ensure_wav(tmp_path)
        voice_conf = predict_voice_confidence(wav_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        for p in {tmp_path, wav_path}:
            if not p:
                continue
            try:
                os.remove(p)
            except OSError:
                pass

    return jsonify({"voice_confidence": voice_conf})