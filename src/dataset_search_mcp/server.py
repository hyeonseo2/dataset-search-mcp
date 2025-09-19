from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import string
import subprocess
import time
import typing as T
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from huggingface_hub import list_datasets, HfApi
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from fastmcp import FastMCP
from typing import Union


# ---------- MCP server bootstrap ----------
mcp = FastMCP("dataset-search-mcp")


# ---------- Utilities ----------
SAFE_TIMEOUT = 20
UA = {"User-Agent": "DatasetSearchMCP/0.1 (+python)"}


VALID_SOURCES = {"huggingface", "zenodo", "kaggle"}

def normalize_sources(sources) -> set[str]:
    """Accept list/tuple/set OR string ('hf, zenodo' / '["hf","zenodo"]') and normalize."""
    if sources is None or sources == []:
        return VALID_SOURCES

    vals: list[str] = []
    if isinstance(sources, (list, tuple, set)):
        vals = [str(x).strip().lower() for x in sources]
    elif isinstance(sources, str):
        s = sources.strip()
        # Try JSON list first
        try:
            loaded = json.loads(s)
            if isinstance(loaded, (list, tuple)):
                vals = [str(x).strip().lower() for x in loaded]
            else:
                vals = []
        except Exception:
            # Fallback: split by comma/space
            vals = [v.strip().lower() for v in re.split(r"[,\s]+", s) if v.strip()]
    else:
        vals = []

    alias = {"hf": "huggingface", "huggingface": "huggingface",
             "zenodo": "zenodo", "kg": "kaggle", "kaggle": "kaggle"}

    norm = [alias.get(v, v) for v in vals]
    picked = {v for v in norm if v in VALID_SOURCES}
    return picked or VALID_SOURCES

def safe_get(url: str, params: dict | None = None, timeout: int = SAFE_TIMEOUT, retries: int = 2):
    """GET with small retry/backoff for flaky endpoints."""
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            if i == retries:
                raise
            time.sleep(1.25 * (i + 1))


def to_dt_str(x) -> str:
    """Coerce datetimes/strings to YYYY-MM-DD; fallback to first 10 chars."""
    if not x:
        return ""
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")
    s = str(x)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.replace("Z", ""), fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return s[:10]


def tokenize(s: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    s = (s or "").lower()
    for ch in string.punctuation:
        s = s.replace(ch, " ")
    return [w for w in s.split() if w]


# ---------- Shared model ----------
@dataclass
class Row:
    source: str
    id: str
    title: str
    description: str
    updated: str
    url: str
    download_url: str
    formats: list[str]


# ---------- Hugging Face ----------
def search_hf(query: str, limit: int = 40) -> list[Row]:
    """List datasets via HF hub; enrich with dataset card when available."""
    out: list[Row] = []
    api = HfApi()
    try:
        ds_list = list_datasets(search=query, limit=limit)
    except Exception:
        return out

    for d in ds_list:
        ds_id = getattr(d, "id", "") or ""
        url = f"https://huggingface.co/datasets/{ds_id}"
        updated = to_dt_str(getattr(d, "lastModified", None) or getattr(d, "updated_at", None))
        desc, fmts = "", []

        try:
            info = api.dataset_info(ds_id, timeout=15)
            card = getattr(info, "cardData", None) or {}
            if isinstance(card, dict):
                desc = (card.get("description") or "")[:2000]
            updated = to_dt_str(getattr(info, "lastModified", None) or getattr(info, "updated_at", None)) or updated
        except Exception:
            pass

        out.append(Row("huggingface", ds_id, ds_id, desc, updated, url, "", fmts))
    return out


# ---------- Zenodo ----------
def search_zenodo(query: str, limit: int = 40) -> list[Row]:
    """Search Zenodo datasets via public API."""
    base = "https://zenodo.org/api/records"
    r = safe_get(base, params={"q": query, "type": "dataset", "size": limit})
    hits = (r.json().get("hits", {}) or {}).get("hits", []) if r is not None else []
    out: list[Row] = []

    for h in hits:
        md = h.get("metadata", {}) or {}
        title = md.get("title") or h.get("title") or ""
        desc = re.sub(r"<[^>]+>", " ", html.unescape(md.get("description") or "")).strip()[:2000]
        url = (h.get("links", {}) or {}).get("html", "") or ""
        files = h.get("files") or []
        fmts = list({(f.get("type") or f.get("mimetype") or "").split("/")[-1] for f in files if f})
        dl = files[0].get("links", {}).get("self", "") if files else ""
        upd = to_dt_str(h.get("updated"))
        out.append(Row("zenodo", str(h.get("id") or ""), title, desc, upd, url, dl, [f for f in fmts if f]))
    return out


# ---------- Kaggle (API → CLI fallback) ----------
def ensure_kaggle_credentials():
    """If env vars exist, write ~/.kaggle/kaggle.json with secure permissions."""
    path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(path):
        return
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (user and key):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"username": user, "key": key}, f)
    os.chmod(path, 0o600)


def kaggle_available() -> bool:
    """True if env creds are set or kaggle.json exists."""
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or os.path.exists(cred_path)


def search_kaggle(query: str, limit: int = 40) -> list[Row]:
    """Try Kaggle API; if empty/fails, fallback to CLI --csv."""
    rows: list[Row] = []
    if not kaggle_available():
        return rows

    ensure_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        try:
            api_res = api.dataset_list(search=query, page=1)  # slice later
        except TypeError:
            api_res = []

        if api_res:
            for d in api_res[:limit]:
                try:
                    m = api.dataset_view(d.ref)
                    desc = (getattr(m, "description", "") or "").strip()
                    upd = to_dt_str(getattr(m, "lastUpdated", None))
                except Exception:
                    desc, upd = "", ""
                fmts = []
                try:
                    files = api.dataset_list_files(d.ref).files
                    for f in files:
                        ext = (f.name.split(".")[-1] if "." in f.name else "").lower()
                        if ext:
                            fmts.append(ext)
                    fmts = sorted(set(fmts))
                except Exception:
                    pass
                url = f"https://www.kaggle.com/datasets/{d.ref}"
                rows.append(Row("kaggle", d.ref, d.title or d.ref, desc, upd, url, url, fmts))
            return rows
    except Exception:
        pass

    # CLI fallback
    try:
        cli = subprocess.run(
            ["kaggle", "datasets", "list", "-s", query, "--csv", "-p", "1", "-r", str(max(20, min(100, limit)))],
            capture_output=True, text=True
        )
        if cli.returncode == 0 and cli.stdout.strip():
            f = io.StringIO(cli.stdout)
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i >= limit:
                    break
                title = r.get("title") or ""
                url = r.get("url") or ""
                ref = "/".join(url.rstrip("/").split("/")[-2:]) if "/datasets/" in url else url
                rows.append(Row("kaggle", ref, title, (r.get("subtitle") or "").strip(),
                                (r.get("lastUpdated") or "")[:10], url, url, []))
    except Exception:
        pass

    return rows


# ---------- Ranking ----------
def rank(query: str, rows: list[Row]) -> list[dict]:
    """BM25 + fuzzy + slight recency weight; returns sorted list of dicts."""
    if not rows:
        return []
    docs = [" ".join([r.title, r.description]) for r in rows]
    bm = BM25Okapi([tokenize(d) for d in docs])
    qs = tokenize(query)
    bm_scores = bm.get_scores(qs)
    mx = max(bm_scores) if len(bm_scores) > 0 else 1.0
    out: list[dict] = []

    for i, r in enumerate(rows):
        fz = fuzz.token_set_ratio(query, f"{r.title} {r.description}") / 100.0
        rec = 0.0
        try:
            if r.updated:
                days = (datetime.utcnow() - datetime.strptime(r.updated, "%Y-%m-%d")).days
                rec = max(0.0, 1.0 - min(days, 365) / 365.0)
        except Exception:
            pass
        score = 0.6 * (bm_scores[i] / (mx + 1e-9)) + 0.35 * fz + 0.05 * rec
        row = asdict(r)
        row["score"] = round(float(score), 4)
        out.append(row)

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ---------- MCP tools ----------
@mcp.tool
def search_datasets(
    query: str,
    sources: Union[list[str], str, None] = None,  # <-- here
    limit: int = 40,
    format_filter: str | None = None,
) -> list[dict]:
    """
    Search public datasets (Hugging Face / Zenodo / (optional) Kaggle).

    Args:
      - query: free-text idea/topic description
      - sources: subset of ["huggingface","zenodo","kaggle"] (default: all)
      - limit: per-source max items before ranking
      - format_filter: e.g., "csv", "json" (applies only if formats are known)
    Returns:
      - ranked list of dataset dicts with score (descending)
    """
    srcs = normalize_sources(sources)
    rows: list[Row] = []

    if "huggingface" in srcs:
        rows += search_hf(query, limit)
    if "zenodo" in srcs:
        rows += search_zenodo(query, limit)
    if "kaggle" in srcs:
        rows += search_kaggle(query, limit)

    if format_filter:
        ff = format_filter.lower()
        rows = [r for r in rows if any(ff in (f or "").lower() for f in (r.formats or []))]

    # dedupe by (source, id)
    uniq: dict[tuple[str, str], Row] = {}
    for r in rows:
        key = (r.source, r.id)
        if key not in uniq:
            uniq[key] = r

    return rank(query, list(uniq.values()))


@mcp.tool
def starter_code(
    source: str,
    id: str,
    url: str = "",
    download_url: str = "",
    formats: list[str] | None = None,
) -> str:
    """
    Generate a minimal, Colab-friendly starter snippet for the selected dataset.

    Args:
      - source: "huggingface" | "zenodo" | "kaggle"
      - id: dataset id (e.g., HF owner/name, Kaggle owner/slug, Zenodo record id)
      - url: landing page (optional)
      - download_url: direct file URL when available (enables CSV quickload)
      - formats: known file extensions (optional)
    """
    fmts = [f.lower() for f in (formats or [])]
    header = f"# Starter code\n# Source: {source}\n# ID: {id}\n# Landing: {url}\n"
    common = """
# !pip install pandas requests
import os, io, requests, pandas as pd
"""

    if any("csv" in f for f in fmts) and download_url:
        return header + common + f"""# Direct CSV download
csv_url = "{download_url}"
r = requests.get(csv_url, timeout=30); r.raise_for_status()
df = pd.read_csv(io.BytesIO(r.content))
print(df.head(), df.shape)
"""

    if source == "huggingface":
        return header + common + f"""# Load via Hugging Face Datasets
# !pip install datasets
from datasets import load_dataset
ds = load_dataset("{id}")
print(ds)
"""

    if source == "kaggle":
        return header + common + f"""# Download via Kaggle API
# !pip install kaggle
# Requires KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.dataset_download_files("{id}", path="data", unzip=True)
print("Downloaded to ./data")
"""

    return header + common + f"""# Refer to the landing page for usage/download instructions.
print("Open:", "{url}")
"""


def main():
    # Run the MCP server over stdio. Most clients (Claude Desktop, Cursor, etc.) can spawn it.
    mcp.run()


if __name__ == "__main__":
    main()
