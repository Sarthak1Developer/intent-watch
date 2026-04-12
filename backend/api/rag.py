from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class RagSource:
    id: str
    type: str
    message: str
    timestamp: str | None
    severity: str | None
    camera: str | None
    snapshot_url: str | None


class _Index:
    def __init__(self) -> None:
        self.fingerprint: Tuple[int, str] | None = None
        self.texts: list[str] = []
        self.sources: list[RagSource] = []
        self.emb: np.ndarray | None = None


_index = _Index()
_lock = threading.Lock()


def _alerts_fingerprint(alerts: list[dict]) -> Tuple[int, str]:
    if not alerts:
        return (0, "")
    last_id = str(alerts[-1].get("id") or "")
    return (len(alerts), last_id)


def _format_alert_doc(a: dict) -> Tuple[str, RagSource]:
    atype = str(a.get("type") or "Alert")
    ts = a.get("timestamp")
    msg = str(a.get("message") or "")
    sev = a.get("severity")
    cam = a.get("camera")
    snap = a.get("snapshot_url")
    aid = str(a.get("id") or "")
    content = "\n".join(
        [
            f"Type: {atype}",
            f"Time: {ts}",
            f"Severity: {sev}",
            f"Camera: {cam}",
            f"Message: {msg}",
        ]
    )
    return (
        content,
        RagSource(
            id=aid,
            type=atype,
            message=msg,
            timestamp=str(ts) if ts is not None else None,
            severity=str(sev) if sev is not None else None,
            camera=str(cam) if cam is not None else None,
            snapshot_url=str(snap) if snap is not None else None,
        ),
    )


def _try_embed(texts: list[str]) -> np.ndarray | None:
    """Return embeddings matrix [n, d] or None if embedding deps are unavailable."""
    model_name = (os.getenv("INTENTWATCH_RAG_EMBED_MODEL") or "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
    except Exception:
        return None


def _lexical_score(query: str, doc: str) -> float:
    q = {t for t in query.lower().split() if len(t) >= 3}
    if not q:
        return 0.0
    d = {t for t in doc.lower().split() if len(t) >= 3}
    if not d:
        return 0.0
    inter = q.intersection(d)
    return float(len(inter)) / float(len(q))


def rebuild_index_from_alerts(alerts: list[dict]) -> None:
    """Best-effort rebuild of the in-memory retrieval index."""
    docs: list[str] = []
    sources: list[RagSource] = []
    for a in alerts:
        d, s = _format_alert_doc(a)
        docs.append(d)
        sources.append(s)

    emb = _try_embed(docs)
    fp = _alerts_fingerprint(alerts)
    _index.fingerprint = fp
    _index.texts = docs
    _index.sources = sources
    _index.emb = emb


def _retrieve(question: str, *, k: int = 5) -> list[RagSource]:
    k = max(1, min(int(k), 12))

    texts = _index.texts
    sources = _index.sources
    if not texts or not sources:
        return []

    emb = _index.emb
    if emb is not None:
        q_emb = _try_embed([question])
        if q_emb is None:
            emb = None
        else:
            sims = (emb @ q_emb[0]).astype(np.float32)
            idx = np.argsort(-sims)[:k]
            return [sources[int(i)] for i in idx]

    # Fallback: lexical overlap
    scored = [(i, _lexical_score(question, t)) for i, t in enumerate(texts)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [sources[i] for i, s in scored[:k] if s > 0]
    return top


def _http_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout_s: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def _generate_answer(question: str, sources: list[RagSource]) -> str:
    """Generate an answer using OpenAI/Ollama if configured; otherwise return extractive context."""
    context = "\n\n".join(
        [
            f"[{i+1}] {s.type} | {s.timestamp} | {s.message} | camera={s.camera} | severity={s.severity}"
            for i, s in enumerate(sources)
        ]
    ).strip()

    provider = (os.getenv("INTENTWATCH_RAG_PROVIDER") or "").strip().lower()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    ollama_url = (os.getenv("INTENTWATCH_OLLAMA_URL") or "http://localhost:11434/api/generate").strip()

    if not provider:
        if openai_key:
            provider = "openai"
        elif ollama_url:
            provider = "ollama"
        else:
            provider = "none"

    if provider == "openai" and openai_key:
        model = (os.getenv("INTENTWATCH_OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
        prompt = (
            "You are IntentWatch assistant. Answer using ONLY the provided alert context. "
            "If the context is insufficient, say what is missing. Keep it concise.\n\n"
            f"ALERT CONTEXT:\n{context}\n\nQUESTION: {question}"
        )
        try:
            resp = _http_json(
                "https://api.openai.com/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You answer questions about surveillance alerts."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                headers={"Authorization": f"Bearer {openai_key}"},
                timeout_s=45.0,
            )
            content = (
                ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
                if isinstance(resp, dict)
                else None
            )
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass

    if provider == "ollama" and ollama_url:
        model = (os.getenv("INTENTWATCH_OLLAMA_MODEL") or "llama3.1").strip() or "llama3.1"
        prompt = (
            "Answer using ONLY this alert context. If unsure, say you are unsure.\n\n"
            f"ALERT CONTEXT:\n{context}\n\nQUESTION: {question}"
        )
        try:
            resp = _http_json(
                ollama_url,
                {"model": model, "prompt": prompt, "stream": False},
                timeout_s=60.0,
            )
            out = resp.get("response") if isinstance(resp, dict) else None
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception:
            pass

    # No LLM configured: return extractive context.
    if not context:
        return "No alert context available yet. Generate some alerts first, then ask again."
    return "Based on recent alerts:\n\n" + context


def answer_question(alerts: list[dict], question: str, *, k: int = 5) -> tuple[str, list[RagSource]]:
    """High-level API: update index if needed, retrieve sources, then generate an answer."""
    q = str(question or "").strip()
    if not q:
        return ("Missing question", [])

    with _lock:
        fp = _alerts_fingerprint(alerts)
        if _index.fingerprint != fp:
            # Rebuild outside hot path as best-effort; keep it simple for hackathon.
            rebuild_index_from_alerts(alerts)

        sources = _retrieve(q, k=k)

        # If retrieval found nothing but we do have alerts, fall back to the most recent
        # items from the provided alerts list (not just the index).
        if not sources and alerts:
            tail_alerts = alerts[-max(1, min(int(k), len(alerts))):]
            recent_sources: list[RagSource] = []
            for a in reversed(tail_alerts):
                try:
                    _doc, src = _format_alert_doc(a)
                    recent_sources.append(src)
                except Exception:
                    continue
            sources = recent_sources

    answer = _generate_answer(q, sources)
    return (answer, sources)
