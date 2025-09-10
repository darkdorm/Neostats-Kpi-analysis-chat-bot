import os
import json
import numpy as np
from typing import List, Dict
from config.config import CONFIG
from openai import OpenAI
from groq import Groq

INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize clients
_openai_client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY")) if CONFIG.get("OPENAI_API_KEY") else None
_groq_client = Groq(api_key=CONFIG.get("GROQ_API_KEY")) if CONFIG.get("GROQ_API_KEY") else None

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Get embeddings for a list of texts.
    Uses OpenAI if LLM_PROVIDER=openai, else Groq (or fallback).
    """
    provider = CONFIG.get("LLM_PROVIDER", "openai")
    results = []

    if provider == "openai":
        if not _openai_client:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        for txt in texts:
            resp = _openai_client.embeddings.create(input=txt, model=model)
            results.append(resp.data[0].embedding)
        return results

    elif provider == "groq":
        # NOTE: Groq API (Sep 2024) does not expose embeddings.
        # For now, we’ll fake by hashing vectors (demo purpose).
        # In real project: use SentenceTransformers or OpenAI just for embeddings.
        import hashlib
        for txt in texts:
            # deterministic pseudo-vector from hash
            h = hashlib.sha256(txt.encode()).digest()
            vec = np.frombuffer(h[:256], dtype=np.uint8).astype(np.float32)
            results.append(vec / np.linalg.norm(vec))  # normalize
        return results

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER={provider}. Use 'openai' or 'groq'.")

def save_index(path_prefix: str, docs: List[Dict]):
    """Save vectors + metadata to disk."""
    vecs = np.array([d["vector"] for d in docs], dtype=np.float32)
    metas = [{"id": d["id"], "text": d["text"], "meta": d.get("meta", {})} for d in docs]
    np.savez_compressed(f"{INDEX_DIR}/{path_prefix}.npz", vectors=vecs)
    with open(f"{INDEX_DIR}/{path_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

def load_index(path_prefix: str):
    """returns (vectors ndarray, metas list) or (None, None)"""
    np_path = f"{INDEX_DIR}/{path_prefix}.npz"
    js_path = f"{INDEX_DIR}/{path_prefix}.json"
    if not (os.path.exists(np_path) and os.path.exists(js_path)):
        return None, None
    arr = np.load(np_path)["vectors"]
    with open(js_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return arr, metas

def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray):
    q = np.array(query_vec, dtype=np.float32)
    M = matrix.astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-10
    M_norm = np.linalg.norm(M, axis=1) + 1e-10
    sims = (M @ q) / (M_norm * q_norm)
    return sims

def query_index(path_prefix: str, query: str, top_k: int = 5, embedding_model="text-embedding-3-small"):
    """Returns top_k docs: list of dicts {id, text, meta, score}"""
    vecs, metas = load_index(path_prefix)
    if vecs is None:
        return []

    q_vec = get_embeddings([query], model=embedding_model)[0]
    sims = cosine_similarity_matrix(np.array(q_vec), vecs)
    top_idx = list(np.argsort(-sims)[: top_k])
    results = []
    for i in top_idx:
        results.append({
            "id": metas[i]["id"],
            "text": metas[i]["text"],
            "meta": metas[i].get("meta", {}),
            "score": float(sims[i])
        })
    return results
