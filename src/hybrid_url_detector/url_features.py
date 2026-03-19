from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract


SUSPICIOUS_TLDS = {
    "zip",
    "mov",
    "click",
    "top",
    "xyz",
    "work",
    "gq",
    "tk",
    "ml",
    "ga",
    "cf",
    "country",
}

KEYWORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "paypal",
    "free",
]


IPV4_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|[0-1]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[0-1]?\d?\d)"
)

_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(text)
    ent = 0.0
    for c in counts.values():
        p = c / length
        ent -= p * math.log2(p)
    return float(ent)


def _count_special(url: str) -> int:
    return sum(1 for ch in url if not ch.isalnum())


def _has_double_slash_redirect(url: str) -> int:
    # `http://a.com//http://b.com` style patterns
    parsed = urlparse(url)
    rest = url
    if parsed.scheme:
        # remove "scheme://"
        prefix = f"{parsed.scheme}://"
        if url.lower().startswith(prefix):
            rest = url[len(prefix) :]
    return int("//" in rest)


def _num_subdomains(domain: str) -> int:
    ext = _TLD_EXTRACT(domain)
    if not ext.domain and not ext.suffix:
        return 0
    if not ext.subdomain:
        return 0
    return len([p for p in ext.subdomain.split(".") if p])


@dataclass(frozen=True)
class FeatureSpec:
    include_keyword_flags: bool = True
    include_entropy: bool = True


def extract_features(urls: Sequence[str], spec: FeatureSpec | None = None) -> pd.DataFrame:
    spec = spec or FeatureSpec()
    rows: List[dict] = []

    for url in urls:
        url_str = "" if url is None else str(url)
        url_l = url_str.lower()
        parsed = urlparse(url_str if "://" in url_str else f"http://{url_str}")
        domain = parsed.netloc
        path = parsed.path or ""

        ext = _TLD_EXTRACT(domain)
        tld = (ext.suffix or "").split(".")[-1] if ext.suffix else ""

        count_digits = sum(ch.isdigit() for ch in url_str)
        url_len = len(url_str)
        special_count = _count_special(url_str)

        row = {
            "url_length": url_len,
            "domain_length": len(domain),
            "path_length": len(path),
            "count_dots": url_str.count("."),
            "count_hyphens": url_str.count("-"),
            "count_digits": count_digits,
            "count_special_chars": special_count,
            "pct_digits": (count_digits / url_len) if url_len else 0.0,
            "pct_special_chars": (special_count / url_len) if url_len else 0.0,
            "has_ip": int(bool(IPV4_RE.search(url_str))),
            "has_https": int(url_l.startswith("https://")),
            "has_at_symbol": int("@" in url_str),
            "has_double_slash_redirect": _has_double_slash_redirect(url_str),
            "num_subdomains": _num_subdomains(domain),
            "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        }

        if spec.include_entropy:
            row["entropy_url"] = shannon_entropy(url_l)
            row["entropy_domain"] = shannon_entropy(domain.lower())

        if spec.include_keyword_flags:
            for kw in KEYWORDS:
                row[f"kw_{kw}"] = int(kw in url_l)

        rows.append(row)

    return pd.DataFrame(rows).fillna(0.0)


def features_and_labels(df: pd.DataFrame, url_col: str = "url", label_col: str = "label") -> tuple[pd.DataFrame, np.ndarray]:
    x = extract_features(df[url_col].astype(str).tolist())
    y = df[label_col].astype(int).to_numpy()
    return x, y
