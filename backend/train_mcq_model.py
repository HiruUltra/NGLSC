# train_mcq_model.py
# =====================================================
# MCQ Question + AnswerOption -> Correct/Wrong Pipeline
#
# Dataset style:
#   - One row per MCQ
#   - Columns:
#       question
#       correct_answer
#       option_2, option_3, option_4, option_5   (or similar)
#
# This script:
#   * Expands each MCQ into multiple samples:
#       text = "question [SEP] candidate_answer"
#       label = 1 if candidate_answer is correct, else 0
#
#   * Splits DATA BY QUESTION (no leakage)
#   * Tries 4 models:
#       LogisticRegression, LinearSVC,
#       RandomForestClassifier, MultinomialNB
#   * Selects best model by f1_weighted
#   * Saves into outputs/<timestamp>/:
#       - classification_metrics.csv / .png
#       - classification_report_<model>.csv / .png
#       - confusion_matrix_<model>.png
#       - best_model.joblib
#       - best_model_predictions.csv
#       - summary.json, README.txt
# =====================================================

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


# ----------------- Helpers -----------------

def slugify(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s.strip())
    return re.sub(r"_{2,}", "_", s).strip("_").lower()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_table_as_png(df: pd.DataFrame, out_path: str, title: str = ""):
    """
    Save a pandas DataFrame as a simple table image.
    """
    df_disp = df.copy()
    for col in df_disp.columns:
        if pd.api.types.is_numeric_dtype(df_disp[col]):
            df_disp[col] = df_disp[col].round(4)

    n_rows, n_cols = df_disp.shape
    fig_h = max(4, 0.5 * n_rows)
    fig_w = max(6, 0.8 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_disp.astype(str).values,
        colLabels=df_disp.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    if title:
        ax.set_title(title, pad=12, fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, title, out_path):
    """
    Plot normalized confusion matrix as heatmap.
    """
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------- Data prep -----------------

def load_mcq_raw(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p.resolve()}")

    try:
        df = pd.read_csv(p)
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="latin1")

    return df


def expand_to_pairs(
    df: pd.DataFrame,
    question_col: str,
    correct_col: str,
    option_cols: list[str],
) -> pd.DataFrame:
    """
    Convert wide MCQ table to pairwise samples:
      question + candidate_answer -> label(0/1)
    """
    df = df.copy()
    df = df.dropna(subset=[question_col, correct_col])
    df[question_col] = df[question_col].astype(str).str.strip()
    df[correct_col] = df[correct_col].astype(str).str.strip()

    # Drop fully empty questions or answers
    df = df[(df[question_col] != "") & (df[correct_col] != "")].copy()
    df = df.reset_index(drop=True)
    df["qid"] = np.arange(len(df))  # unique question ID

    rows = []

    for _, row in df.iterrows():
        qid = row["qid"]
        q_text = str(row[question_col]).strip()
        correct_ans = str(row[correct_col]).strip()

        # Positive: correct answer
        text_pos = f"{q_text} [SEP] {correct_ans}"
        rows.append(
            {
                "qid": qid,
                "pair_text": text_pos,
                "label": 1,
                "question": q_text,
                "candidate_answer": correct_ans,
                "is_correct_option": True,
            }
        )

        # Negatives: all other options
        for col in option_cols:
            val = row.get(col, None)
            if pd.isna(val):
                continue
            cand = str(val).strip()
            if cand == "" or cand == correct_ans:
                continue
            text_neg = f"{q_text} [SEP] {cand}"
            rows.append(
                {
                    "qid": qid,
                    "pair_text": text_neg,
                    "label": 0,
                    "question": q_text,
                    "candidate_answer": cand,
                    "is_correct_option": False,
                }
            )

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        raise ValueError("After expansion, no pairwise samples were created. Check your option columns.")

    print(f"[info] Expanded to {len(pairs_df)} pairwise samples "
          f"from {df.shape[0]} questions.")
    print(pairs_df["label"].value_counts())

    return pairs_df


def infer_option_cols(df: pd.DataFrame, question_col: str, correct_col: str) -> list[str]:
    """
    If user didn't specify option columns, try to guess:
    all object/string columns except question & correct.
    """
    candidates = []
    for col in df.columns:
        if col in [question_col, correct_col]:
            continue
        if df[col].dtype == "object":
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "Could not infer option columns automatically. "
            "Please pass them explicitly with --option_cols."
        )

    print(f"[info] Inferred option columns: {candidates}")
    return candidates


# ----------------- Model zoo -----------------

def get_model_zoo() -> dict[str, Pipeline]:
    base_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        stop_words="english",
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "MultinomialNB": MultinomialNB(),
    }

    zoo = {}
    for name, clf in models.items():
        zoo[name] = Pipeline(
            [
                ("tfidf", base_tfidf),
                ("clf", clf),
            ]
        )
    return zoo


# ----------------- Training -----------------

def train_and_evaluate_all(pairs_df: pd.DataFrame, outdir: str):
    """
    Train all models, save:
      - classification_metrics.(csv/png)
      - per-model confusion matrix + classification report
      - best_model.joblib
      - best_model_predictions.csv
    """
    # Split by QUESTION (qid), not per pair
    qids = pairs_df["qid"].unique()
    train_qids, test_qids = train_test_split(
        qids,
        test_size=0.2,
        random_state=42,
    )

    train_df = pairs_df[pairs_df["qid"].isin(train_qids)].reset_index(drop=True)
    test_df = pairs_df[pairs_df["qid"].isin(test_qids)].reset_index(drop=True)

    print(f"[info] Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"[info] Train questions: {len(train_qids)}, Test questions: {len(test_qids)}")

    X_train = train_df["pair_text"].values
    y_train = train_df["label"].values
    X_test = test_df["pair_text"].values
    y_test = test_df["label"].values

    models = get_model_zoo()
    unique_labels = [0, 1]  # wrong / correct

    metrics_rows = {}
    trained_models = {}

    # ---- Train + evaluate each model ----
    for name, model in models.items():
        print(f"\n[info] Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-score : {f1:.4f}")

        metrics_rows[name] = {
            "model": name,
            "accuracy": acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        }

        trained_models[name] = model

        # Confusion matrix + classification report for THIS model
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_path = os.path.join(outdir, f"confusion_matrix_{slugify(name)}.png")
        plot_confusion_matrix(
            cm,
            labels=["wrong (0)", "correct (1)"],
            title=f"Confusion Matrix - {name}",
            out_path=cm_path,
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=unique_labels,
            target_names=["wrong (0)", "correct (1)"],
            zero_division=0,
            output_dict=True,
        )
        rep_df = pd.DataFrame(report).T
        rep_csv = os.path.join(outdir, f"classification_report_{slugify(name)}.csv")
        rep_png = os.path.join(outdir, f"classification_report_{slugify(name)}.png")
        rep_df.to_csv(rep_csv)
        save_table_as_png(rep_df, rep_png, title=f"Classification Report - {name}")

    # ---- Overall metrics table ----
    metrics_df = pd.DataFrame(list(metrics_rows.values()))
    metrics_df = metrics_df.sort_values(
        by="f1_weighted",
        ascending=False,
    ).reset_index(drop=True)

    metrics_csv = os.path.join(outdir, "classification_metrics.csv")
    metrics_png = os.path.join(outdir, "classification_metrics.png")
    metrics_df.to_csv(metrics_csv, index=False)
    save_table_as_png(metrics_df, metrics_png, title="Model Comparison Metrics")
    print(f"\n[info] Saved metrics table to:")
    print(f"  {metrics_csv}")
    print(f"  {metrics_png}")

    # ---- Pick best model ----
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\n[info] Best model by F1_weighted: {best_name}")

    # Save best model
    best_model_path = os.path.join(outdir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[info] Saved best model to: {best_model_path}")

    # Save pair-level predictions for best model
    best_pred = best_model.predict(X_test)
    pred_df = test_df.copy()
    pred_df["pred_label"] = best_pred
    pred_df["pred_is_correct"] = pred_df["pred_label"] == 1
    pred_csv = os.path.join(outdir, "best_model_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"[info] Saved best model predictions to: {pred_csv}")

    # Simple summary JSON
    summary = {
        "n_pair_samples": int(len(pairs_df)),
        "n_train_samples": int(len(train_df)),
        "n_test_samples": int(len(test_df)),
        "n_questions_total": int(pairs_df["qid"].nunique()),
        "n_train_questions": int(len(train_qids)),
        "n_test_questions": int(len(test_qids)),
        "models_tried": list(models.keys()),
        "best_model": best_name,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # README
    with open(os.path.join(outdir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"MCQ Pairwise Training Artifacts - {datetime.now().isoformat()}\n"
            f"Output directory: {outdir}\n\n"
            "Each sample is: 'question [SEP] candidate_answer' -> label (0/1)\n\n"
            "Files:\n"
            "  - classification_metrics.csv / .png : model comparison table (pair-level)\n"
            "  - classification_report_<model>.csv / .png : per-model binary reports\n"
            "  - confusion_matrix_<model>.png : per-model confusion matrices\n"
            "  - best_model.joblib : best performing model (TF-IDF + classifier)\n"
            "  - best_model_predictions.csv : test samples with predicted vs true labels\n"
            "  - summary.json : simple JSON summary\n"
        )

    print("\n✅ Done. All artifacts saved.")


# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MCQ pairwise model: (question, candidate_answer) -> correct/wrong."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="mcq.csv",
        help="Path to MCQ dataset CSV.",
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default="question",
        help="Name of the question text column.",
    )
    parser.add_argument(
        "--correct_col",
        type=str,
        default="correct_answer",
        help="Name of the correct answer column.",
    )
    parser.add_argument(
        "--option_cols",
        nargs="*",
        default=None,
        help=(
            "Names of wrong option columns, e.g. option_2 option_3 option_4 option_5. "
            "If omitted, will infer from other string columns."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: outputs/<timestamp>/).",
    )

    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("outputs", stamp)
    ensure_dir(outdir)

    print(f"[info] Output directory: {outdir}")

    df_raw = load_mcq_raw(args.csv)

    for col in [args.question_col, args.correct_col]:
        if col not in df_raw.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Available columns: {list(df_raw.columns)}"
            )

    if args.option_cols is None or len(args.option_cols) == 0:
        option_cols = infer_option_cols(df_raw, args.question_col, args.correct_col)
    else:
        missing = [c for c in args.option_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"Some option_cols not found in CSV: {missing}. "
                f"Available columns: {list(df_raw.columns)}"
            )
        option_cols = args.option_cols
        print(f"[info] Using explicit option columns: {option_cols}")

    pairs_df = expand_to_pairs(df_raw, args.question_col, args.correct_col, option_cols)
    train_and_evaluate_all(pairs_df, outdir)


if __name__ == "__main__":
    main()

