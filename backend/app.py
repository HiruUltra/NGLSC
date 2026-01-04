
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# app.py
# ======================================================
# MCQ + ER/Flowchart API
#
# MCQ:
#   GET  /quiz      -> returns N random questions from mcq.csv
#   POST /submit    -> grades answers using MCQ ML model (best_model.joblib) if available
#
# ER/Flow:
#   GET  /er/question -> random target class (not OTHER_WRONG)
#   POST /er/submit   -> upload image + question_id, checks using BEST image model
#
# Optional training endpoints (safe fixed args):
#   POST /admin/train/mcq
#   POST /admin/train/er
# Enabled only if ENABLE_TRAIN_ENDPOINTS=1
# ======================================================

import os
import sys
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

# ---- MCQ ML model ----
import joblib

# ---- ER Image model ----
import torch
import torch.nn as nn
from PIL import Image
from process_diagram import image_stream_to_svg  # OpenCV->SVG prototype
from torchvision import models, transforms


# ======================================================
# CONFIG
# ======================================================

# ------------ MCQ (Question bank) ------------
CSV_PATH = os.environ.get("MCQ_CSV_PATH", "mcq.csv")
QUESTION_COL = os.environ.get("MCQ_QUESTION_COL", "question")
CORRECT_COL  = os.environ.get("MCQ_CORRECT_COL", "correct_answer")
OPTION_COLS = None  # auto infer if None

# ------------ MCQ (Trained model) ------------
MCQ_MODEL_PATH = os.environ.get("MCQ_MODEL_PATH", "outputs/latest_mcq/best_model.joblib")
MCQ_THRESHOLD = float(os.environ.get("MCQ_THRESHOLD", "0.5"))  # only used if confidence exists

# ------------ ER (BEST Model-Zoo output) ------------
# ✅ IMPORTANT: Your screenshot shows latest run is outputs/20260104_001541
ER_RUN_ROOT = os.environ.get("ER_RUN_ROOT", "outputs/20260104_001541")

ER_BEST_MODEL_PATH = os.environ.get("ER_BEST_MODEL_PATH", f"{ER_RUN_ROOT}/BEST/best_model.pth")
ER_BEST_BACKBONE_PATH = os.environ.get("ER_BEST_BACKBONE_PATH", f"{ER_RUN_ROOT}/BEST/best_backbone.txt")

# question map (optional)
ER_QUESTIONS_PATH = os.environ.get("ER_QUESTIONS_PATH", "data/er_questions.xlsx")

# optional confidence threshold:
# if confidence < threshold => treat as OTHER_WRONG
ER_CONF_THRESHOLD = float(os.environ.get("ER_CONF_THRESHOLD", "0.0"))  # example set 0.60 if you want

# ------------ Optional: training endpoints ------------
ENABLE_TRAIN_ENDPOINTS = os.environ.get("ENABLE_TRAIN_ENDPOINTS", "0") == "1"
TRAIN_ER_SCRIPT = os.environ.get("TRAIN_ER_SCRIPT", "train_er_model.py")
TRAIN_MCQ_SCRIPT = os.environ.get("TRAIN_MCQ_SCRIPT", "train_mcq_model.py")

TRAIN_LOCK = Lock()


# ======================================================
# Helpers
# ======================================================

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def normalize_id(x: str) -> str:
    return str(x).strip().upper().replace(" ", "").replace("-", "").replace("_", "")

def path_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


# ======================================================
# MCQ: Load dataset
# ======================================================

def load_mcq_dataset():
    p = Path(CSV_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find mcq.csv at: {p.resolve()}")

    try:
        df = pd.read_csv(p)
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="latin1")

    if QUESTION_COL not in df.columns or CORRECT_COL not in df.columns:
        raise ValueError(
            f"CSV must contain '{QUESTION_COL}' and '{CORRECT_COL}'. "
            f"Available: {list(df.columns)}"
        )

    df[QUESTION_COL] = df[QUESTION_COL].astype(str).str.strip()
    df[CORRECT_COL]  = df[CORRECT_COL].astype(str).str.strip()
    df = df[(df[QUESTION_COL] != "") & (df[CORRECT_COL] != "")].reset_index(drop=True)
    df["qid"] = df.index
    return df

def infer_option_cols(df):
    global OPTION_COLS
    if OPTION_COLS is not None:
        return OPTION_COLS

    candidates = []
    for col in df.columns:
        if col in [QUESTION_COL, CORRECT_COL, "qid"]:
            continue
        if df[col].dtype == "object":
            candidates.append(col)

    if not candidates:
        raise ValueError("Could not infer option columns. Set OPTION_COLS manually.")
    OPTION_COLS = candidates
    print("[info] Inferred option columns:", OPTION_COLS)
    return OPTION_COLS

def get_options_for_row(row):
    correct = str(row[CORRECT_COL]).strip()
    options = []
    if correct:
        options.append(correct)

    for col in OPTION_COLS:
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        txt = str(val).strip()
        if txt and txt not in options:
            options.append(txt)

    random.shuffle(options)
    return options


# ======================================================
# MCQ: Load trained model (joblib)
# ======================================================

mcq_model = None

def load_mcq_model():
    global mcq_model
    p = Path(MCQ_MODEL_PATH)
    if not p.exists():
        print(f"[warn] MCQ model not found at {p.resolve()} -> using direct match fallback")
        mcq_model = None
        return None
    try:
        mcq_model = joblib.load(p)
        print(f"[info] Loaded MCQ model: {p}")
        return mcq_model
    except Exception as e:
        print(f"[warn] Failed to load MCQ model: {e} -> fallback direct match")
        mcq_model = None
        return None

def mcq_pair_text(question: str, candidate: str) -> str:
    return f"{str(question).strip()} [SEP] {str(candidate).strip()}"

def sigmoid(x: float) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0

def mcq_predict_is_correct(question: str, candidate: str):
    """
    Returns:
      is_correct(bool), confidence(float|None), method(str)
    """
    if mcq_model is None:
        return None, None, "no_model"

    text = mcq_pair_text(question, candidate)

    # label
    try:
        pred = int(mcq_model.predict([text])[0])
        is_correct = (pred == 1)
    except Exception:
        return None, None, "model_error"

    # confidence (optional)
    conf = None
    if hasattr(mcq_model, "predict_proba"):
        try:
            proba = mcq_model.predict_proba([text])[0]
            if len(proba) >= 2:
                conf = float(proba[1])
        except Exception:
            conf = None
    elif hasattr(mcq_model, "decision_function"):
        try:
            score = mcq_model.decision_function([text])[0]
            conf = float(sigmoid(float(score)))
        except Exception:
            conf = None

    return is_correct, conf, "mcq_ml"


# ======================================================
# ER: Build model by backbone
# ======================================================

def read_best_backbone() -> str:
    p = Path(ER_BEST_BACKBONE_PATH)
    if not p.exists():
        print(f"[warn] best_backbone.txt missing at {p.resolve()} -> fallback vgg16")
        return "vgg16"
    return p.read_text(encoding="utf-8").strip()

def build_model_by_backbone(backbone: str, num_classes: int):
    b = backbone.lower().strip()

    if b == "vgg16":
        try:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.vgg16(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model

    if b == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if b == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if b == "mobilenet_v3_large":
        try:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.mobilenet_v3_large(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if b == "densenet121":
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


er_model = None
er_class_index = None
er_backbone = None
er_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

er_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_class_index_fallback(run_root: str, backbone: str):
    """
    Try these in order:
      1) <run>/<backbone>/class_index.json
      2) <run>/<backbone>/summary.json -> classes
    """
    class_index_path = Path(run_root) / backbone / "class_index.json"
    if class_index_path.exists():
        with open(class_index_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return {int(k): v for k, v in j.items()}, str(class_index_path)

    summary_path = Path(run_root) / backbone / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            sj = json.load(f)
        classes = sj.get("classes", None)
        if isinstance(classes, list) and len(classes) > 0:
            return {i: str(name) for i, name in enumerate(classes)}, str(summary_path) + " (classes)"

    return None, None

def load_er_model_and_classes():
    global er_model, er_class_index, er_backbone

    model_path = Path(ER_BEST_MODEL_PATH)
    if not model_path.exists():
        print(f"[warn] ER BEST model not found: {model_path.resolve()}")
        er_model, er_class_index, er_backbone = None, None, None
        return None, None, None

    backbone = read_best_backbone()

    class_index, source = load_class_index_fallback(ER_RUN_ROOT, backbone)
    if class_index is None:
        print(f"[warn] Could not load class mapping for backbone='{backbone}'. "
              f"Expected class_index.json OR summary.json->classes in {Path(ER_RUN_ROOT) / backbone}")
        er_model, er_class_index, er_backbone = None, None, None
        return None, None, None

    num_classes = len(class_index)

    model = build_model_by_backbone(backbone, num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(er_device)

    er_model, er_class_index, er_backbone = model, class_index, backbone
    print(f"[info] Loaded ER BEST model: backbone={backbone}, classes={num_classes}, device={er_device}")
    print(f"[info] Class mapping loaded from: {source}")
    return er_model, er_class_index, er_backbone

def load_er_question_map():
    p = Path(ER_QUESTIONS_PATH)
    if not p.exists():
        return {}

    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    id_col, text_col = None, None
    for c in df.columns:
        cl = c.lower()
        if cl in ["question_id", "id", "qid"]:
            id_col = c
        if cl in ["question_text", "question_tex", "question"]:
            text_col = c

    if id_col is None or text_col is None:
        return {}

    return {str(r[id_col]).strip(): str(r[text_col]).strip() for _, r in df.iterrows()}

def predict_er_image(img: Image.Image):
    if er_model is None or er_class_index is None:
        return None, None

    x = er_transform(img.convert("RGB")).unsqueeze(0).to(er_device)
    with torch.no_grad():
        logits = er_model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    conf = float(conf.item())
    idx = int(idx.item())
    pred_class = er_class_index.get(idx, "UNKNOWN")

    if ER_CONF_THRESHOLD > 0 and conf < ER_CONF_THRESHOLD:
        pred_class = "OTHER_WRONG"

    return pred_class, conf

def get_random_er_question(er_question_map: dict):
    if er_class_index is None:
        return None

    classes = list(er_class_index.values())
    valid = [c for c in classes if str(c).upper() != "OTHER_WRONG"]
    if not valid:
        valid = classes

    qid = random.choice(valid)
    qtext = er_question_map.get(qid, f"Draw the correct diagram for {qid} (ER/Flowchart).")
    return {"question_id": qid, "question_text": qtext}


# ======================================================
# Flask init
# ======================================================

app = Flask(__name__)
CORS(app)

# Load everything at startup
df_mcq = load_mcq_dataset()
infer_option_cols(df_mcq)

load_mcq_model()
er_question_map = load_er_question_map()
load_er_model_and_classes()


# ======================================================
# ROUTES: DEBUG
# ======================================================

@app.route("/debug/er", methods=["GET"])
def debug_er():
    backbone = None
    try:
        backbone = read_best_backbone()
    except Exception:
        backbone = None

    return jsonify({
        "ER_RUN_ROOT": ER_RUN_ROOT,
        "ER_BEST_MODEL_PATH": ER_BEST_MODEL_PATH,
        "ER_BEST_BACKBONE_PATH": ER_BEST_BACKBONE_PATH,
        "exists": {
            "run_root": path_exists(ER_RUN_ROOT),
            "best_model": path_exists(ER_BEST_MODEL_PATH),
            "best_backbone_txt": path_exists(ER_BEST_BACKBONE_PATH),
            "backbone_folder": path_exists(str(Path(ER_RUN_ROOT) / str(backbone))) if backbone else False,
            "class_index_json": path_exists(str(Path(ER_RUN_ROOT) / str(backbone) / "class_index.json")) if backbone else False,
            "summary_json": path_exists(str(Path(ER_RUN_ROOT) / str(backbone) / "summary.json")) if backbone else False,
        },
        "loaded": {
            "er_model": bool(er_model is not None),
            "er_class_index": bool(er_class_index is not None),
            "er_backbone": er_backbone,
            "device": str(er_device),
        }
    })


# ======================================================
# ROUTES: MCQ
# ======================================================

@app.route("/quiz", methods=["GET"])
def get_quiz():
    try:
        n = int(request.args.get("n", 10))
    except ValueError:
        n = 10
    n = max(1, min(n, len(df_mcq)))

    sample_df = df_mcq.sample(n=n, random_state=None)
    questions = []
    for _, row in sample_df.iterrows():
        qid = int(row["qid"])
        questions.append({
            "id": qid,
            "question": str(row[QUESTION_COL]),
            "options": get_options_for_row(row),
        })
    return jsonify({"questions": questions})


@app.route("/submit", methods=["POST"])
def submit_answers():
    data = request.get_json(force=True, silent=True) or {}
    answers = data.get("answers", [])
    user_id = data.get("user_id", None)

    if not isinstance(answers, list) or len(answers) == 0:
        return jsonify({"error": "No answers provided"}), 400

    results = []
    score = 0

    for item in answers:
        try:
            qid = int(item["id"])
            selected = str(item["selected"]).strip()
        except Exception:
            continue

        if qid not in df_mcq["qid"].values:
            continue

        row = df_mcq.loc[df_mcq["qid"] == qid].iloc[0]
        question_text = str(row[QUESTION_COL])
        correct_answer = str(row[CORRECT_COL]).strip()

        ml_is_correct, ml_conf, ml_method = mcq_predict_is_correct(question_text, selected)

        if ml_is_correct is None:
            is_correct = (selected == correct_answer)
            method = "direct_match"
            confidence = None
        else:
            if ml_conf is not None:
                is_correct = (ml_conf >= MCQ_THRESHOLD)
                confidence = ml_conf
                method = f"{ml_method}_threshold"
            else:
                is_correct = bool(ml_is_correct)
                confidence = None
                method = ml_method

        if is_correct:
            score += 1

        results.append({
            "id": qid,
            "question": question_text,
            "selected": selected,
            "correct_answer": correct_answer,
            "is_correct": bool(is_correct),
            "method": method,
            "confidence": confidence,
        })

    total = len(results)

    if total > 0:
        log_path = "user_responses.csv"
        now = datetime.utcnow().isoformat()
        log_rows = []
        for r in results:
            log_rows.append({
                "timestamp_utc": now,
                "user_id": user_id,
                "qid": r["id"],
                "question": r["question"],
                "selected": r["selected"],
                "correct_answer": r["correct_answer"],
                "is_correct": r["is_correct"],
                "method": r["method"],
                "confidence": r["confidence"],
            })
        log_df = pd.DataFrame(log_rows)
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            log_df.to_csv(log_path, index=False, encoding="utf-8")

    return jsonify({"score": score, "total": total, "results": results})


# ======================================================
# ROUTES: ER / FLOWCHART
# ======================================================

@app.route("/er/question", methods=["GET"])
def er_get_question():
    if er_model is None or er_class_index is None:
        return jsonify({"error": "ER/Flowchart model not loaded"}), 500

    q = get_random_er_question(er_question_map)
    if q is None:
        return jsonify({"error": "No ER classes available"}), 500
    return jsonify(q)


@app.route("/er/submit", methods=["POST"])
def er_submit():
    if er_model is None or er_class_index is None:
        return jsonify({"error": "ER/Flowchart model not loaded"}), 500

    question_id = request.form.get("question_id", "").strip()
    file = request.files.get("image")

    if not question_id:
        return jsonify({"error": "Missing question_id"}), 400
    if file is None:
        return jsonify({"error": "Missing image file (key: 'image')"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    pred_class, conf = predict_er_image(img)
    if pred_class is None:
        return jsonify({"error": "Prediction failed"}), 500

    is_correct = (normalize_id(pred_class) == normalize_id(question_id))

    return jsonify({
        "model_backbone": er_backbone,
        "confidence_threshold": ER_CONF_THRESHOLD,
        "question_id": question_id,
        "predicted_class": pred_class,
        "is_correct": bool(is_correct),
        "confidence": float(conf) if conf is not None else None,
    })


@app.route("/er/convert", methods=["POST"])
def er_convert():
    """Convert an uploaded diagram image (handwritten/photo) into a cleaned SVG.
    Returns JSON: {"svg": "<svg>...</svg>"}
    """
    file = request.files.get("image")

    if file is None:
        return jsonify({"error": "Missing image file (key: 'image')"}), 400

    try:
        svg = image_stream_to_svg(file.stream)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    return jsonify({"svg": svg})


# ======================================================
# OPTIONAL: TRAIN ENDPOINTS (ONLY IF ENABLE_TRAIN_ENDPOINTS=1)
# ======================================================

@app.route("/admin/train/mcq", methods=["POST"])
def admin_train_mcq():
    if not ENABLE_TRAIN_ENDPOINTS:
        return jsonify({"error": "Training endpoints disabled. Set ENABLE_TRAIN_ENDPOINTS=1"}), 403

    with TRAIN_LOCK:
        outdir = f"outputs/mcq_runs/{now_stamp()}"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, TRAIN_MCQ_SCRIPT,
            "--csv", CSV_PATH,
            "--question_col", QUESTION_COL,
            "--correct_col", CORRECT_COL,
            "--outdir", outdir
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            ok = (p.returncode == 0)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        model_path = str(Path(outdir) / "best_model.joblib")
        if ok and Path(model_path).exists():
            global MCQ_MODEL_PATH
            MCQ_MODEL_PATH = model_path
            load_mcq_model()

        return jsonify({
            "ok": ok,
            "returncode": p.returncode,
            "outdir": outdir,
            "mcq_model_path": MCQ_MODEL_PATH,
            "stdout_tail": p.stdout[-2000:],
            "stderr_tail": p.stderr[-2000:],
        }), (200 if ok else 500)


@app.route("/admin/train/er", methods=["POST"])
def admin_train_er():
    if not ENABLE_TRAIN_ENDPOINTS:
        return jsonify({"error": "Training endpoints disabled. Set ENABLE_TRAIN_ENDPOINTS=1"}), 403

    with TRAIN_LOCK:
        data_dir = request.json.get("data_dir", "data/er_images") if request.is_json else "data/er_images"
        outdir = f"outputs/er_runs/{now_stamp()}"
        Path(outdir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, TRAIN_ER_SCRIPT,
            "--data_dir", data_dir,
            "--model_zoo",
            "--outdir", outdir
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            ok = (p.returncode == 0)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        if ok and (Path(outdir) / "BEST" / "best_model.pth").exists():
            global ER_RUN_ROOT, ER_BEST_MODEL_PATH, ER_BEST_BACKBONE_PATH
            ER_RUN_ROOT = outdir
            ER_BEST_MODEL_PATH = f"{ER_RUN_ROOT}/BEST/best_model.pth"
            ER_BEST_BACKBONE_PATH = f"{ER_RUN_ROOT}/BEST/best_backbone.txt"
            load_er_model_and_classes()

        return jsonify({
            "ok": ok,
            "returncode": p.returncode,
            "outdir": outdir,
            "er_run_root": ER_RUN_ROOT,
            "stdout_tail": p.stdout[-2000:],
            "stderr_tail": p.stderr[-2000:],
        }), (200 if ok else 500)


# ======================================================
# ROOT
# ======================================================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "MCQ + ER/Flowchart API",
        "mcq": {
            "csv_path": CSV_PATH,
            "model_loaded": bool(mcq_model is not None),
            "model_path": MCQ_MODEL_PATH,
            "threshold": MCQ_THRESHOLD,
        },
        "er": {
            "model_loaded": bool(er_model is not None),
            "run_root": ER_RUN_ROOT,
            "backbone": er_backbone,
            "device": str(er_device),
            "threshold": ER_CONF_THRESHOLD,
            "best_model_path": ER_BEST_MODEL_PATH,
        },
        "train_endpoints_enabled": ENABLE_TRAIN_ENDPOINTS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
