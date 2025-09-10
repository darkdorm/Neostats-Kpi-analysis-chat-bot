# utils/rag_utils.py
import pandas as pd
import uuid
from typing import List, Dict
from models.embeddings import get_embeddings, save_index, query_index

def read_table(file_stream, filename: str) -> pd.DataFrame:
    """
    Accepts uploaded file (BytesIO or similar) and returns DataFrame.
    Supports CSV and Excel.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file_stream)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_stream)
    else:
        raise ValueError("Unsupported file type. Upload CSV or Excel.")
    return df

def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums

def simple_kpi_extraction(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build simple KPIs: totals, means, growth for numeric columns.
    Returns dict keyed by KPI name with details.
    """
    kpis = {}
    numeric_cols = detect_numeric_columns(df)
    for col in numeric_cols:
        series = df[col].dropna()
        total = float(series.sum())
        mean = float(series.mean())
        last = float(series.iloc[-1]) if len(series) >= 1 else None
        first = float(series.iloc[0]) if len(series) >= 1 else None
        growth = None
        if first is not None and first != 0:
            growth = (last - first) / abs(first)
        kpis[col] = {
            "total": total,
            "mean": mean,
            "first": first,
            "last": last,
            "growth_fraction": growth
        }
    return kpis

def build_chunks_from_kpis(kpis: Dict[str, Dict]) -> List[Dict]:
    """
    For each KPI produce a human-readable chunk summarizing the metric.
    """
    docs = []
    for col, d in kpis.items():
        uid = str(uuid.uuid4())
        growth_pct = None
        if d["growth_fraction"] is not None:
            growth_pct = round(d["growth_fraction"] * 100, 2)
        text = f"KPI: {col}\nTotal: {d['total']}\nAverage: {d['mean']}\nFirst: {d['first']}\nLast: {d['last']}\nGrowth%: {growth_pct}"
        docs.append({"id": uid, "text": text, "meta": {"column": col}})
    return docs

def build_index_for_dataframe(df, index_name: str = "default_index") -> Dict:
    """
    Extract KPIs, build chunks, compute embeddings and save index.
    Returns metadata about index (num docs, index_name).
    """
    kpis = simple_kpi_extraction(df)
    chunks = build_chunks_from_kpis(kpis)
    texts = [c["text"] for c in chunks]
    vectors = get_embeddings(texts)
    for i, c in enumerate(chunks):
        c["vector"] = vectors[i]
    save_index(index_name, chunks)
    return {"index_name": index_name, "num_docs": len(chunks)}

def rag_query(index_name: str, user_query: str, top_k: int = 5):
    """
    Wrapper around models.embeddings.query_index
    """
    hits = query_index(index_name, user_query, top_k=top_k)
    return hits
