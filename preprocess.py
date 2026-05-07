"""
preprocess.py - Project B: News Source Classification (Fox News vs NBC News)

Contract (from eval_project_b.py):
    prepare_data(csv_path: str) -> (X, y)
        X: list of cleaned headline strings
        y: list[int], with FoxNews=1 and NBC=0
"""

from __future__ import annotations

import html
import numbers
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd


LABEL_FOX = 1
LABEL_NBC = 0
LABEL_NAMES = {LABEL_FOX: "FoxNews", LABEL_NBC: "NBC"}

_HEADLINE_COLS: Sequence[str] = (
    "headline",
    "headline_clean",
    "scraped_headline",
    "alternative_headline",
    "title",
    "text",
)
_URL_COLS: Sequence[str] = ("url", "link", "URL", "article_url")
_LABEL_COLS: Sequence[str] = ("label", "source", "publisher", "y")


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _label_from_url(url: object) -> Optional[int]:
    if not isinstance(url, str) or not url:
        return None
    u = url.lower()
    if "foxnews.com" in u or "foxbusiness.com" in u:
        return LABEL_FOX
    if "nbcnews.com" in u or "today.com" in u or "msnbc.com" in u:
        return LABEL_NBC
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return None
    if "fox" in host:
        return LABEL_FOX
    if "nbc" in host:
        return LABEL_NBC
    return None


def _label_from_string(v: object) -> Optional[int]:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "fox", "foxnews", "fox news", "fox_news"}:
            return LABEL_FOX
        if s in {"0", "nbc", "nbcnews", "nbc news", "nbc_news"}:
            return LABEL_NBC
        if "fox" in s:
            return LABEL_FOX
        if "nbc" in s:
            return LABEL_NBC
    elif isinstance(v, numbers.Number) and not pd.isna(v):
        fv = float(v)
        if fv in (0.0, 1.0):
            return int(fv)
    return None


_RE_HTML = re.compile(r"<[^>]+>")
_RE_FOX_SUFFIX = re.compile(r"\s*[\|\-–—]\s*Fox News.*$", re.IGNORECASE)
_RE_NBC_SUFFIX = re.compile(r"\s*[\|\-–—]\s*NBC News.*$", re.IGNORECASE)
_RE_FOX_PREFIX = re.compile(
    r"^\s*(?:FOX\s+NEWS\s+(?:ALERT|EXCLUSIVE|POLL|POWER\s+RANKINGS)"
    r"|Fox\s+News(?:\s+(?:Exclusive|Poll|Power\s+Rankings))?)\s*:\s*",
    re.IGNORECASE,
)
_RE_NBC_PREFIX = re.compile(r"^\s*NBC\s+News\s*:\s*", re.IGNORECASE)
_RE_FOX_NATION = re.compile(r"\bFOX\s+Nation\b", re.IGNORECASE)
_RE_WS = re.compile(r"\s+")


def _clean_text(t: object) -> str:
    if not isinstance(t, str):
        return ""
    t = html.unescape(unicodedata.normalize("NFKC", t))
    t = _RE_HTML.sub(" ", t)
    t = _RE_FOX_SUFFIX.sub("", t)
    t = _RE_NBC_SUFFIX.sub("", t)
    t = _RE_FOX_PREFIX.sub("", t)
    t = _RE_NBC_PREFIX.sub("", t)
    t = _RE_FOX_NATION.sub("Nation", t)
    return _RE_WS.sub(" ", t).strip()


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path)

    headline_col = _find_col(df, _HEADLINE_COLS)
    url_col = _find_col(df, _URL_COLS)
    label_col = _find_col(df, _LABEL_COLS)

    if headline_col is None:
        raise ValueError(
            f"No headline column found. Looked for {list(_HEADLINE_COLS)}; "
            f"got columns {list(df.columns)}"
        )
    if url_col is None and label_col is None:
        raise ValueError(
            f"Need a URL column ({list(_URL_COLS)}) or label column "
            f"({list(_LABEL_COLS)}) to derive y; got {list(df.columns)}"
        )

    X: List[str] = []
    y: List[int] = []
    for _, row in df.iterrows():
        text = _clean_text(row[headline_col])
        if not text:
            continue

        label: Optional[int] = None
        if url_col is not None:
            label = _label_from_url(row[url_col])
        if label is None and label_col is not None:
            label = _label_from_string(row[label_col])
        if label is None:
            continue

        X.append(text)
        y.append(label)

    return X, y


if __name__ == "__main__":
    import sys

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "project-resources/Newsheadlines/url_with_headlines.csv"
    )
    X, y = prepare_data(csv_path)
    fox = sum(1 for v in y if v == LABEL_FOX)
    nbc = sum(1 for v in y if v == LABEL_NBC)
    print(f"n={len(X)}  FoxNews={fox}  NBC={nbc}")
    for i in range(min(3, len(X))):
        print(f"[{LABEL_NAMES[y[i]]}] {X[i]}")
