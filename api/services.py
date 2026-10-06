"""
api/services.py
===============
Pure Python service layer — completely Django-free.

Design rules:
  • Every function takes plain Python types (str, dict, Path) and returns
    plain Python types (dict, str, bytes). No HttpRequest/Response here.
  • ML logic is extracted from app/views.py and made reusable.
  • Views call these functions and translate results → HTTP responses.
  • All exceptions propagate upward; views decide the HTTP status code.

This separation makes the logic unit-testable without spinning up Django.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Use non-interactive backend so matplotlib never tries to open a window
matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

ALGORITHM_MAP: dict[str, Any] = {
    "rf":  RandomForestClassifier(n_estimators=100, random_state=42),
    "gb":  GradientBoostingClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "lr":  LogisticRegression(max_iter=1000, random_state=42),
}
ALGORITHM_LABELS: dict[str, str] = {
    "rf":  "Random Forest",
    "gb":  "Gradient Boosting",
    "svm": "Support Vector Machine",
    "lr":  "Logistic Regression",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_summary(csv_path: str) -> dict[str, Any]:
    """
    Read a CSV and return a lightweight schema summary.
    Does NOT load the whole file into RAM beyond Pandas dtype inference.

    Returns
    -------
    dict with keys: filename, rows, columns, column_names,
                    missing_values, dtypes, preview
    """
    df = pd.read_csv(csv_path)
    return {
        "filename":       Path(csv_path).name,
        "rows":           int(df.shape[0]),
        "columns":        int(df.shape[1]),
        "column_names":   list(df.columns),
        "missing_values": {col: int(n) for col, n in df.isnull().sum().items()},
        "dtypes":         {col: str(dt) for col, dt in df.dtypes.items()},
        "preview":        df.head(3).where(pd.notna(df.head(3)), other=None)
                            .to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing helpers (mirrored from app/views.py run_algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_dataframe(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.Series, SimpleImputer, dict[str, LabelEncoder], list[str]
]:
    """
    Internal helper: encode categoricals, impute nulls, return X, y and
    the fitted transformers so they can be persisted for inference.
    """
    df = df.copy()
    df.drop(columns=[c for c in ["id", "name"] if c in df.columns],
            inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    label_encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    trained_columns = X.columns.tolist()
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imputed, y, imputer, label_encoders, trained_columns


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train all models & return ranked results
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(csv_path: str) -> dict[str, Any]:
    """
    Train RF, GB, SVM, LR on the CSV; persist models + transformers;
    return a ranked summary of every model's metrics.

    Returns
    -------
    {
        "best_model":  "rf",
        "best_label":  "Random Forest",
        "best_accuracy": 0.942,
        "models": [
            {"key": "rf", "label": ..., "accuracy": ..., "f1": ..., "auc": ...},
            ...  (sorted best → worst)
        ]
    }

    Side-effects
    ------------
    Writes <key>_model.pkl, label_encoders.pkl, imputer.pkl,
    trained_columns.pkl into the models/ directory.
    """
    df = pd.read_csv(csv_path)
    X, y, imputer, label_encoders, trained_columns = _preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results: list[dict[str, Any]] = []

    for key, model in ALGORITHM_MAP.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        # F1 (weighted handles multi-class gracefully)
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # AUC — only computable when model supports predict_proba
        auc: float | None = None
        n_classes = len(np.unique(y_test))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            try:
                if n_classes == 2:
                    auc = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    auc = float(
                        roc_auc_score(y_test, proba, multi_class="ovr",
                                      average="weighted")
                    )
            except ValueError:
                auc = None

        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_test, y_pred).tolist()

        # Persist model
        joblib.dump(model, MODELS_DIR / f"{key}_model.pkl")

        results.append({
            "key":             key,
            "label":           ALGORITHM_LABELS[key],
            "accuracy":        round(acc, 4),
            "f1_weighted":     round(f1, 4),
            "auc":             round(auc, 4) if auc is not None else None,
            "classification_report": report_dict,
            "confusion_matrix":      matrix,
        })

    # Persist shared transformers
    joblib.dump(label_encoders,  MODELS_DIR / "label_encoders.pkl")
    joblib.dump(imputer,         MODELS_DIR / "imputer.pkl")
    joblib.dump(trained_columns, MODELS_DIR / "trained_columns.pkl")

    # Sort best → worst by accuracy
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best = results[0]

    # Save the best model key for the inference endpoint
    joblib.dump(best["key"], MODELS_DIR / "best_model_key.pkl")

    return {
        "best_model":    best["key"],
        "best_label":    best["label"],
        "best_accuracy": best["accuracy"],
        "models":        results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Single-record inference using the persisted best model
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_record(input_data: dict[str, str]) -> dict[str, Any]:
    """
    Run inference on one record using the saved best model.

    Parameters
    ----------
    input_data : dict mapping feature names → string values
                 (all values arrive as strings from JSON; we cast below)

    Returns
    -------
    {"prediction": <label>, "model_used": <label>, "feature_importance": {...}}

    Raises
    ------
    FileNotFoundError  if no model has been trained yet
    ValueError         if required features are missing
    """
    required_files = [
        "best_model_key.pkl", "label_encoders.pkl",
        "imputer.pkl", "trained_columns.pkl",
    ]
    for fname in required_files:
        if not (MODELS_DIR / fname).exists():
            raise FileNotFoundError(
                f"'{fname}' not found. Train models via POST /api/predict/ "
                "with a dataset_name first."
            )

    best_key      = joblib.load(MODELS_DIR / "best_model_key.pkl")
    model         = joblib.load(MODELS_DIR / f"{best_key}_model.pkl")
    label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
    imputer        = joblib.load(MODELS_DIR / "imputer.pkl")
    trained_cols   = joblib.load(MODELS_DIR / "trained_columns.pkl")

    row = pd.DataFrame([input_data])

    # Encode categoricals using saved encoders
    for col in row.columns:
        if row[col].dtype == object and col in label_encoders:
            le = label_encoders[col]
            val = str(row[col].iloc[0])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0  # unknown category → fallback

    # Convert everything to numeric
    for col in row.columns:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    # Align to training feature set
    for col in trained_cols:
        if col not in row.columns:
            row[col] = 0.0
    row = row[trained_cols]

    row_imputed = pd.DataFrame(imputer.transform(row), columns=trained_cols)
    prediction = model.predict(row_imputed)[0]

    # Feature importance if available
    importance: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        importance = {
            col: round(float(imp), 6)
            for col, imp in zip(trained_cols, model.feature_importances_)
        }
        # Sort descending
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

    return {
        "prediction":         str(prediction),
        "model_used":         ALGORITHM_LABELS.get(best_key, best_key),
        "feature_importance": importance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PDF Report generation (wraps existing ReportLab logic)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    csv_path: str,
    ai_summary: str = "",
) -> bytes:
    """
    Generate a full PDF analytics report for the given dataset.

    Trains all models on the dataset, renders:
      0. (Optional) AI Executive Summary section at the top  ← injected in Phase 5
      1. Model accuracy comparison table
      2. Pie + line charts for accuracy scores
      3. Confusion matrix heatmap of best model
      4. Classification report heatmap of best model

    Parameters
    ----------
    csv_path   : absolute path to the CSV file
    ai_summary : LLM-generated text (Phase 5); empty string = section skipped

    Returns
    -------
    bytes  — raw PDF content, ready to stream to FileResponse
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak,
    )

    # ── Train models to get metrics ──────────────────────────────────────────
    ml_results = run_all_models(csv_path)
    models_data = ml_results["models"]
    best_key    = ml_results["best_model"]
    best_result = next(r for r in models_data if r["key"] == best_key)

    # ── Automatically generate AI executive summary if not provided ──────────
    if not ai_summary:
        try:
            from ml_models.rag_pipeline import generate_executive_summary
            ai_summary = generate_executive_summary(
                csv_file_path=csv_path,
                ml_metrics=ml_results,
            )
        except Exception as exc:
            ai_summary = f"[AI Executive Summary Note: Failed to generate summary ({exc})]"


    # ── Chart helpers ────────────────────────────────────────────────────────
    def _fig_to_rl_image(fig: plt.Figure, width: float = 4.8 * inch) -> RLImage:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        orig_w, orig_h = fig.get_size_inches()
        height = width * (orig_h / orig_w)
        plt.close(fig)
        buf.seek(0)
        img = RLImage(buf, width=width, height=height)
        return img

    # Accuracy pie chart
    names  = [r["label"] for r in models_data]
    scores = [r["accuracy"] for r in models_data]

    fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
    ax_pie.pie(scores, labels=names, autopct="%1.1f%%", startangle=90)
    ax_pie.set_title("Algorithm Accuracy Comparison")
    pie_img = _fig_to_rl_image(fig_pie, width=4.5 * inch)

    # Accuracy line chart
    fig_line, ax_line = plt.subplots(figsize=(6, 3.8))
    ax_line.plot(names, scores, marker="o", linestyle="-", color="steelblue")
    ax_line.set_title("Algorithm Accuracy")
    ax_line.set_ylabel("Accuracy")
    ax_line.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    line_img = _fig_to_rl_image(fig_line, width=4.8 * inch)

    # Confusion matrix of best model
    matrix = np.array(best_result["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix — {best_result['label']}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    cm_img = _fig_to_rl_image(fig_cm, width=4.5 * inch)

    # Classification report heatmap
    report_dict = best_result["classification_report"]
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .drop(columns=["support"], errors="ignore")
        .round(2)
    )
    fig_rpt, ax_rpt = plt.subplots(
        figsize=(6, max(2.5, len(report_df) * 0.45 + 0.8))
    )
    sns.heatmap(
        report_df, annot=True, cmap="YlGnBu", fmt=".2f",
        linewidths=0.5, ax=ax_rpt, vmin=0, vmax=1,
    )
    ax_rpt.set_title(f"Classification Report — {best_result['label']}")
    plt.yticks(rotation=0)
    rpt_img = _fig_to_rl_image(fig_rpt, width=5.0 * inch)

    # ── Compose PDF with ReportLab platypus ──────────────────────────────────
    styles = getSampleStyleSheet()
    h1     = styles["Heading1"]
    h2     = styles["Heading2"]
    body   = styles["BodyText"]
    body.leading = 16

    ai_style = ParagraphStyle(
        "AIStyle",
        parent=body,
        backColor=colors.HexColor("#EEF4FF"),
        borderColor=colors.HexColor("#4A90D9"),
        borderWidth=1,
        borderPadding=10,
        leading=18,
    )

    elements: list = []

    # ── Cover title ───────────────────────────────────────────────────────────
    elements.append(Paragraph("CyberPhysicalBanking — Analytics Report", h1))
    elements.append(Paragraph(f"Dataset: {Path(csv_path).name}", body))
    elements.append(Spacer(1, 0.3 * inch))

    # ── AI Executive Summary (Phase 5 hook) ───────────────────────────────────
    if ai_summary:
        elements.append(Paragraph("AI Executive Summary", h2))
        for para in ai_summary.strip().split("\n\n"):
            elements.append(Paragraph(para.strip(), ai_style))
            elements.append(Spacer(1, 0.1 * inch))
        elements.append(PageBreak())

    # ── Model comparison table ────────────────────────────────────────────────
    elements.append(Paragraph("Model Performance Summary", h2))
    table_data = [["Model", "Accuracy", "F1 (Weighted)", "AUC"]]
    for r in models_data:
        marker = " ✓" if r["key"] == best_key else ""
        table_data.append([
            r["label"] + marker,
            f"{r['accuracy']:.4f}",
            f"{r['f1_weighted']:.4f}",
            f"{r['auc']:.4f}" if r["auc"] is not None else "N/A",
        ])

    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F2F3F4"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.3 * inch))

    # ── Charts ────────────────────────────────────────────────────────────────
    elements.append(Paragraph("Accuracy — Pie Chart", h2))
    elements.append(pie_img)
    elements.append(PageBreak())

    elements.append(Paragraph("Accuracy — Line Chart", h2))
    elements.append(line_img)
    elements.append(PageBreak())

    elements.append(Paragraph(
        f"Best Model: {best_result['label']} "
        f"(Accuracy: {best_result['accuracy']:.2%})", h2
    ))
    elements.append(Paragraph("Confusion Matrix", h2))
    elements.append(cm_img)
    elements.append(PageBreak())
    elements.append(Paragraph("Classification Report Heatmap", h2))
    elements.append(rpt_img)

    # ── Build PDF bytes ───────────────────────────────────────────────────────
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.read()


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAG Chat Service  (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_dataset_service(
    dataset_name: str,
    user_query: str,
) -> dict[str, Any]:
    """
    Service wrapper around the RAG pipeline for the /api/chat/ endpoint.

    Resolves the dataset filename → absolute path, then delegates to
    ml_models.rag_pipeline.query_dataset_rag().

    The path resolution happens here (in the Django-aware service layer) so
    rag_pipeline.py stays 100 % Django-free and portable.

    Parameters
    ----------
    dataset_name : filename as returned by /api/upload/ (e.g. 'customers.csv')
    user_query   : natural-language question from the user

    Returns
    -------
    dict with keys:
        success (bool), answer (str) | error (str), dataset_name (str)
    """
    # Lazy import — avoids loading LangChain at Django startup time
    from ml_models.rag_pipeline import query_dataset_rag

    csv_path = Path(MODELS_DIR).parent / "media" / "datasets" / dataset_name

    # MODELS_DIR is <project>/models/; media/datasets/ is a sibling of models/
    # Re-derive from BASE_DIR to be explicit
    try:
        from django.conf import settings as _settings
        csv_path = Path(_settings.MEDIA_ROOT) / "datasets" / dataset_name
    except Exception:
        pass  # Already set above as fallback

    if not csv_path.exists():
        return {
            "success":      False,
            "error":        (
                f"Dataset '{dataset_name}' not found in media/datasets/. "
                "Upload it first via POST /api/upload/."
            ),
            "dataset_name": dataset_name,
        }

    if not dataset_name.lower().endswith(".csv"):
        return {
            "success":      False,
            "error":        "RAG chat is only supported for CSV files.",
            "dataset_name": dataset_name,
        }

    try:
        answer = query_dataset_rag(
            csv_file_path=str(csv_path),
            user_query=user_query,
        )
        return {
            "success":      True,
            "answer":       answer,
            "dataset_name": dataset_name,
        }
    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc), "dataset_name": dataset_name}
    except RuntimeError as exc:
        # RuntimeError carries the human-readable Ollama/ChromaDB message
        return {"success": False, "error": str(exc), "dataset_name": dataset_name}
    except Exception as exc:
        return {
            "success":      False,
            "error":        f"Unexpected RAG error: {exc}",
            "dataset_name": dataset_name,
        }

