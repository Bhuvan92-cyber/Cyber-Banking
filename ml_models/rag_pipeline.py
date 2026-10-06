"""
ml_models/rag_pipeline.py
==========================
RAG (Retrieval-Augmented Generation) pipeline for natural-language
querying of tabular banking datasets.

Architecture decision — why NOT embed raw CSV rows:
  • A 10 000-row CSV ≈ 2–5 M tokens: unusable with any context window.
  • Embedding every cell creates high-dimensional noise with no semantic gain.
  • Industry best practice: extract a ~300-token "Dataset Profile" (schema +
    stats + sample rows), embed THAT, and use it as LLM context.
  • ChromaDB stores the profile keyed by dataset hash so re-parsing only
    happens once per unique file, not per query.

Module contract:
  • Zero Django knowledge. No HttpRequest/Response/models imports.
  • Accepts plain Python types (str) and returns plain Python types (str).
  • Every LangChain/Ollama failure is caught and re-raised as a plain
    RuntimeError with a human-readable message — the view decides HTTP code.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Runtime configuration (read from Django settings via lazy import) ──────────

def _get_rag_config() -> dict[str, Any]:
    """
    Lazily import Django settings or environment variables.
    Provides decoupled configuration for LLM, embeddings, and ChromaDB.
    """
    try:
        from django.conf import settings
        llm_model = getattr(settings, "OLLAMA_LLM_MODEL", getattr(settings, "OLLAMA_MODEL", "llama3"))
        embedding_model = getattr(settings, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        base_dir = str(getattr(settings, "BASE_DIR", Path(__file__).resolve().parent.parent))
        chroma_host = getattr(settings, "CHROMA_HOST", None)
        chroma_port = getattr(settings, "CHROMA_PORT", 8000)
        chroma_persist_dir = getattr(settings, "CHROMA_PERSIST_DIRECTORY", "chroma_db")
    except Exception:
        llm_model = os.environ.get("OLLAMA_LLM_MODEL", os.environ.get("OLLAMA_MODEL", "llama3"))
        embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        base_dir = str(Path(__file__).resolve().parent.parent)
        chroma_host = os.environ.get("CHROMA_HOST", "").strip() or None
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000")) if os.environ.get("CHROMA_PORT") else 8000
        chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "chroma_db")

    if not os.path.isabs(chroma_persist_dir):
        chroma_persist_dir = os.path.join(base_dir, chroma_persist_dir)

    os.makedirs(chroma_persist_dir, exist_ok=True)

    return {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "base_url": base_url,
        "base_dir": base_dir,
        "chroma_host": chroma_host,
        "chroma_port": chroma_port,
        "chroma_persist_dir": chroma_persist_dir,
    }


def _get_settings() -> tuple[str, str, str]:
    """
    Backward-compatible tuple (llm_model, base_url, base_dir).
    """
    cfg = _get_rag_config()
    return cfg["llm_model"], cfg["base_url"], cfg["base_dir"]


def _chroma_dir() -> str:
    """Return (and create) the persistent ChromaDB directory."""
    cfg = _get_rag_config()
    return cfg["chroma_persist_dir"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Profile Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_profile(csv_path: str) -> str:
    """
    Generate a token-efficient text summary of a CSV dataset.

    Sections produced (≈ 250–400 tokens total):
      • Shape and file name
      • Column names + dtypes
      • Numeric statistics (mean / std / min / max / median)
      • Categorical column value counts (top 5 per column)
      • Missing value counts (non-zero only)
      • First 5 sample rows

    Parameters
    ----------
    csv_path : absolute path to the CSV file

    Returns
    -------
    str : multi-line text profile ready to be embedded or passed as LLM context
    """
    df = pd.read_csv(csv_path)
    filename = Path(csv_path).name
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"=== Dataset Profile: {filename} ===",
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        "",
    ]

    # ── Column schema ─────────────────────────────────────────────────────────
    lines.append("── Column Schema ──")
    for col, dtype in df.dtypes.items():
        null_count = int(df[col].isnull().sum())
        null_note  = f"  [{null_count} missing]" if null_count else ""
        lines.append(f"  {col!r:<30} {str(dtype):<12}{null_note}")
    lines.append("")

    # ── Numeric statistics ────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("── Numeric Column Statistics ──")
        stats = df[numeric_cols].agg(["mean", "std", "min", "median", "max"]).round(4)
        lines.append(stats.to_string())
        lines.append("")

    # ── Categorical value counts (top 5) ─────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        lines.append("── Categorical Column Value Counts (top 5) ──")
        for col in cat_cols[:8]:  # cap at 8 columns to stay token-efficient
            top = df[col].value_counts().head(5)
            top_dict = {str(k): int(v) for k, v in top.items()}
            lines.append(f"  {col!r}: {top_dict}")
        lines.append("")

    # ── Missing values (only if any) ──────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        lines.append("── Missing Values ──")
        for col, count in missing.items():
            pct = count / len(df) * 100
            lines.append(f"  {col!r}: {count} ({pct:.1f}%)")
        lines.append("")

    # ── Sample rows ───────────────────────────────────────────────────────────
    lines.append("── Sample Data (first 5 rows) ──")
    lines.append(df.head(5).to_string(index=False))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChromaDB profile cache
# ─────────────────────────────────────────────────────────────────────────────

def _file_hash(csv_path: str) -> str:
    """
    Compute a stable hash of the CSV file's *content* (not just the name)
    so that if a user re-uploads a different file with the same name the
    cache is properly invalidated.

    Uses first 1 MB + file size as a fast fingerprint (avoids reading huge
    files fully into memory just for hashing).
    """
    h    = hashlib.md5()
    size = os.path.getsize(csv_path)
    with open(csv_path, "rb") as f:
        h.update(f.read(1_048_576))   # first 1 MB
    h.update(str(size).encode())
    return h.hexdigest()


def _get_or_build_profile(
    csv_path: str,
    embeddings,
) -> str:
    """
    Return the cached dataset profile from ChromaDB, or build and cache it.

    Uses a content-hash as the ChromaDB collection name so re-uploads of
    changed files get a fresh profile automatically.
    Always supplies client-side embeddings explicitly to avoid server-side 501 errors.
    """
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    cfg = _get_rag_config()
    collection_name = f"ds_{_file_hash(csv_path)}"

    if cfg["chroma_host"]:
        logger.info(
            "RAG: Connecting to Chroma HTTP server at %s:%s (collection=%s)",
            cfg["chroma_host"], cfg["chroma_port"], collection_name
        )
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            host=cfg["chroma_host"],
            port=cfg["chroma_port"],
        )
    else:
        persist_dir = cfg["chroma_persist_dir"]
        logger.info(
            "RAG: Connecting to persistent Chroma at %s (collection=%s)",
            persist_dir, collection_name
        )
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    # Check if we already stored a profile for this exact file
    existing = vectorstore.similarity_search("dataset profile schema", k=1)
    if existing:
        logger.info("RAG: Cache hit for %s (collection=%s)", csv_path, collection_name)
        return existing[0].page_content

    # Cache miss — build, store, return
    logger.info("RAG: Cache miss — building profile for %s", csv_path)
    profile_text = build_dataset_profile(csv_path)

    doc = Document(
        page_content=profile_text,
        metadata={
            "source":   csv_path,
            "filename": Path(csv_path).name,
            "type":     "dataset_profile",
        },
    )
    vectorstore.add_documents([doc])
    logger.info("RAG: Profile cached in collection=%s", collection_name)
    return profile_text


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_ANALYST_PROMPT = """\
You are an expert Banking Data Analyst AI assistant.
Your role is to answer questions about a dataset using ONLY the information \
provided in the Dataset Profile below.

Rules:
- Base your answer strictly on the profile. Do not invent numbers.
- If the answer cannot be determined from the profile, say so clearly.
- Keep the answer concise, structured, and professional.
- Use bullet points or numbered lists where appropriate.
- If asked about trends, reference the statistics provided.

Dataset Profile:
{context}

User Question: {question}

Analyst Answer:"""

_EXEC_SUMMARY_PROMPT = """\
You are an expert Banking Data Analyst AI. Based on the dataset profile and \
the ML model evaluation metrics provided below, write a concise, \
3-paragraph plain-English Executive Summary for a banking stakeholder.

Paragraph 1: Dataset overview — what data exists, key columns, data quality issues.
Paragraph 2: Model performance — which algorithm performed best, accuracy, F1, AUC, \
and what that means in plain English.
Paragraph 3: Key business insights and recommended next steps for the banking team.

Keep the tone professional and avoid technical jargon. Total length: ~200–300 words.

Dataset Profile:
{context}

ML Model Metrics:
{metrics}

Executive Summary:"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Public API
# ─────────────────────────────────────────────────────────────────────────────

def query_dataset_rag(
    csv_file_path: str,
    user_query: str,
    *,
    timeout_seconds: int = 120,
) -> str:
    """
    Answer a natural-language question about a CSV dataset using RAG.

    Flow:
      1. Build (or retrieve from ChromaDB cache) the dataset profile.
      2. Feed the profile + user question into the analyst prompt template.
      3. Stream the response from the local Ollama LLM.
      4. Return the text answer.

    Parameters
    ----------
    csv_file_path   : absolute path to the uploaded CSV
    user_query      : free-text question from the user
    timeout_seconds : how long to wait for the Ollama response (default 120 s)

    Returns
    -------
    str  — LLM-generated answer

    Raises
    ------
    FileNotFoundError  if csv_file_path does not exist
    RuntimeError       wrapping any LangChain / Ollama failure, with a
                       human-readable message (no stack traces leaked to API)
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(
            f"Dataset not found at path: {csv_file_path!r}. "
            "Upload it first via POST /api/upload/."
        )

    cfg = _get_rag_config()
    llm_model = cfg["llm_model"]
    embedding_model = cfg["embedding_model"]
    ollama_base_url = cfg["base_url"]

    # ── Lazy imports — LangChain is heavy; only pay cost when endpoint is hit ──
    try:
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        from langchain_core.prompts import PromptTemplate
    except ImportError as exc:
        raise RuntimeError(
            "LangChain Ollama packages are not installed. "
            "Run: pip install langchain-ollama langchain-chroma chromadb"
        ) from exc

    # ── Initialise LLM and embeddings ─────────────────────────────────────────
    try:
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Ollama embeddings for model '{embedding_model}': {exc}\n"
            f"• Is Ollama running at {ollama_base_url}?\n"
            f"• Has embedding model '{embedding_model}' been pulled? Run: ollama pull {embedding_model}"
        ) from exc

    llm = OllamaLLM(
        model=llm_model,
        base_url=ollama_base_url,
        temperature=0.2,        # low temp → factual, grounded answers
        timeout=timeout_seconds,
    )

    # ── Retrieve or build profile ─────────────────────────────────────────────
    try:
        context = _get_or_build_profile(csv_file_path, embeddings)
    except Exception as exc:
        exc_str = str(exc)
        if "501" in exc_str or "does not support embeddings" in exc_str:
            raise RuntimeError(
                f"Embedding error: {exc_str}\n"
                f"The model '{embedding_model}' on {ollama_base_url} does not support embeddings. "
                f"Ensure embedding model is pulled via: ollama pull {embedding_model}"
            ) from exc
        elif "connection refused" in exc_str.lower() or "connecterror" in exc_str.lower():
            raise RuntimeError(
                f"Connection error reaching Ollama or ChromaDB: {exc_str}\n"
                f"Ensure Ollama is running at {ollama_base_url}."
            ) from exc
        raise RuntimeError(
            f"Failed to build/retrieve dataset profile: {exc_str}\n"
            f"Verify ChromaDB configuration and that embeddings can be generated."
        ) from exc

    # ── Build and invoke the LangChain chain ─────────────────────────────────
    prompt   = PromptTemplate.from_template(_ANALYST_PROMPT)
    chain    = prompt | llm

    try:
        response: str = chain.invoke({
            "context":  context,
            "question": user_query,
        })
        return response.strip()
    except Exception as exc:
        exc_str = str(exc)
        if "connection refused" in exc_str.lower() or "connecterror" in exc_str.lower():
            raise RuntimeError(
                f"Ollama connection refused at {ollama_base_url}: {exc_str}. "
                "Ensure Ollama is running."
            ) from exc
        raise RuntimeError(
            f"Ollama LLM call failed ({llm_model}): {exc}\n"
            f"• Is Ollama running at {ollama_base_url}?\n"
            f"• Has model '{llm_model}' been pulled? Run: ollama pull {llm_model}"
        ) from exc


def generate_executive_summary(
    csv_file_path: str,
    ml_metrics: Optional[dict] = None,
    *,
    timeout_seconds: int = 180,
) -> str:
    """
    Generate a 3-paragraph plain-English Executive Summary for a PDF report.

    Parameters
    ----------
    csv_file_path : absolute path to the CSV
    ml_metrics    : dict containing model evaluation metrics or results.
    timeout_seconds : Ollama timeout (default 180 s — summaries take longer)

    Returns
    -------
    str  — 3-paragraph executive summary ready to inject into the PDF
    """
    if not os.path.exists(csv_file_path):
        return f"[AI Summary Unavailable: Dataset not found at {csv_file_path}]"

    cfg = _get_rag_config()
    llm_model = cfg["llm_model"]
    embedding_model = cfg["embedding_model"]
    ollama_base_url = cfg["base_url"]

    try:
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        from langchain_core.prompts import PromptTemplate
    except ImportError as exc:
        logger.warning("LangChain Ollama packages not installed: %s", exc)
        return (
            "Executive Summary (Automated Fallback):\n\n"
            "The uploaded dataset was successfully analyzed across key financial and transaction features. "
            "Data profiles and statistical checks were executed across numerical and categorical variables.\n\n"
            "Machine learning models were evaluated on benchmark classification metrics. "
            "For full local LLM narrative summaries, ensure Ollama is installed and running.\n\n"
            "Decision makers are advised to review the confusion matrix and classification reports below."
        )

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base_url)
    llm = OllamaLLM(
        model=llm_model,
        base_url=ollama_base_url,
        temperature=0.3,
        timeout=timeout_seconds,
    )

    # Get (or build) the cached profile
    try:
        context = _get_or_build_profile(csv_file_path, embeddings)
    except Exception as exc:
        logger.warning("Profile retrieval failed: %s", exc)
        context = build_dataset_profile(csv_file_path)

    # Format metrics for the prompt
    if isinstance(ml_metrics, dict):
        if "models" in ml_metrics:
            best = ml_metrics.get("best_label", "Unknown")
            rows = []
            for m in ml_metrics.get("models", []):
                auc_str = f"{m['auc']:.4f}" if m.get("auc") is not None else "N/A"
                rows.append(
                    f"  - {m['label']}: accuracy={m['accuracy']:.4f}, "
                    f"f1={m['f1_weighted']:.4f}, auc={auc_str}"
                )
            metrics_text = (
                f"Best model: {best} (accuracy={ml_metrics.get('best_accuracy', 0):.4f})\n"
                + "\n".join(rows)
            )
        else:
            metrics_text = "\n".join([f"- {k}: {v}" for k, v in ml_metrics.items()])
    elif ml_metrics:
        metrics_text = str(ml_metrics)
    else:
        metrics_text = "Standard classification baseline metrics evaluated."

    prompt = PromptTemplate.from_template(_EXEC_SUMMARY_PROMPT)
    chain  = prompt | llm

    try:
        summary: str = chain.invoke({
            "context": context,
            "metrics": metrics_text,
        })
        return summary.strip()
    except Exception as exc:
        logger.warning("Executive summary generation failed: %s", exc)
        return (
            f"[AI Executive Summary Note: Local LLM narrative could not be generated ({exc}). "
            f"Ensure Ollama is running at {ollama_base_url} with model '{ollama_model}' pulled.]"
        )

