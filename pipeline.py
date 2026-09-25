"""
Business Entity Resolution Pipeline
Amazon ML Challenge 2026

Consolidated, reusable pipeline functions:
  Stage 1: Normalization (with French abbreviation handling)
  Stage 2: Blocking / candidate generation (token + digit + address-token + FAISS)
  Stage 3: Pairwise feature engineering
  Stage 4: Matching classifier (LightGBM)
  Stage 5: Threshold-based grouping + F_0.5 scoring

Run via the notebook that imports this module — see README.md for reproduction steps.
"""

import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    from unidecode import unidecode
except ImportError:
    raise ImportError("Run: pip install unidecode")


# ============================================================
# STAGE 1: NORMALIZATION
# ============================================================

GENERIC_BUSINESS_ABBR = {
    r"\bcorp\b": "corporation",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"\b&\b": "and",
}

ADDRESS_ABBR = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bfl(oo)?r\b": "floor",
    r"\bopp\b": "opposite",
}

# French-specific — kept separate since some tokens are ambiguous (e.g. "ste")
FRENCH_ABBR = {
    r"\bbd\b": "boulevard",
    r"\bblvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\brte\b": "route",
    r"\bpl\b": "place",
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\bsa\b": "societe anonyme",
    r"\bsas\b": "societe par actions simplifiee",
}

STOPWORDS = {"the", "and", "inc", "incorporated", "corporation", "limited", "private",
             "company", "llc", "llp", "ltd", "pvt", "co", "of", "a", "an"}

ADDR_STOPWORDS = {"road", "street", "avenue", "floor", "near", "opposite", "block",
                   "sector", "no", "colony", "nagar", "main", "unit", "apartment"}


def disambiguate_ste(text):
    """'ste' at the START of a token sequence is usually Sainte (Saint);
    mid/end-of-name as a business suffix is usually Societe."""
    tokens = text.split()
    out = []
    for i, tok in enumerate(tokens):
        if tok in ("ste", "sté"):
            out.append("sainte" if i == 0 else "societe")
        else:
            out.append(tok)
    return " ".join(out)


def normalize_text(text, country=None):
    """Lowercase, transliterate, strip punctuation, expand abbreviations."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if country == "France":
        text = disambiguate_ste(text)
        for pat, repl in FRENCH_ABBR.items():
            text = re.sub(pat, repl, text)

    for pat, repl in GENERIC_BUSINESS_ABBR.items():
        text = re.sub(pat, repl, text)
    for pat, repl in ADDRESS_ABBR.items():
        text = re.sub(pat, repl, text)

    return re.sub(r"\s+", " ", text).strip()


def extract_digit_sequences(text):
    """Digit runs (house numbers, PIN/ZIP codes) — order-independent signal."""
    if pd.isna(text) or text is None:
        return set()
    return set(re.findall(r"\d+", str(text)))


def apply_normalization(df):
    """Add norm_name, norm_address, addr_digits columns to a source dataframe."""
    df = df.copy()
    df["norm_name"] = df.apply(lambda r: normalize_text(r["business_name"], r.get("country")), axis=1)
    df["norm_address"] = df.apply(lambda r: normalize_text(r["business_address"], r.get("country")), axis=1)
    df["addr_digits"] = df["business_address"].apply(extract_digit_sequences)
    return df


# ============================================================
# STAGE 2: BLOCKING / CANDIDATE GENERATION
# ============================================================

def build_pruned_token_index(df, text_col, id_col="entity_id", max_ratio=0.01):
    """token -> set of entity_ids, dropping tokens that appear in >max_ratio of records."""
    index = defaultdict(set)
    for eid, text in zip(df[id_col], df[text_col]):
        tokens = set(text.split()) - STOPWORDS
        for tok in tokens:
            if len(tok) >= 3:
                index[tok].add(eid)
    max_size = max(5, int(len(df) * max_ratio))
    pruned = {tok: ids for tok, ids in index.items() if len(ids) <= max_size}
    return pruned, len(index) - len(pruned)


def build_pruned_digit_index(df, id_col="entity_id", max_ratio=0.01, min_digits=2):
    index = defaultdict(set)
    for eid, digits in zip(df[id_col], df["addr_digits"]):
        for d in digits:
            if len(d) >= min_digits:
                index[d].add(eid)
    max_size = max(5, int(len(df) * max_ratio))
    pruned = {d: ids for d, ids in index.items() if len(ids) <= max_size}
    return pruned, len(index) - len(pruned)


def build_pruned_addr_token_index(df, id_col="entity_id", max_ratio=0.01):
    index = defaultdict(set)
    for eid, text in zip(df[id_col], df["norm_address"]):
        tokens = set(text.split()) - STOPWORDS - ADDR_STOPWORDS
        for tok in tokens:
            if len(tok) >= 4:
                index[tok].add(eid)
    max_size = max(5, int(len(df) * max_ratio))
    pruned = {tok: ids for tok, ids in index.items() if len(ids) <= max_size}
    return pruned, len(index) - len(pruned)


def build_country_indexes(df, builder_fn, *args, **kwargs):
    """Run a builder_fn per-country group; returns {country: index}."""
    result = {}
    total_dropped = 0
    for country, sub in df.groupby("country"):
        idx, dropped = builder_fn(sub, *args, **kwargs)
        result[country] = idx
        total_dropped += dropped
    return result, total_dropped


def token_block_candidates(s1_row, index):
    tokens = set(s1_row["norm_name"].split()) - STOPWORDS
    candidates = set()
    for tok in tokens:
        if len(tok) >= 3 and tok in index:
            candidates |= index[tok]
    return candidates


def digit_block_candidates(s1_row, index):
    candidates = set()
    for d in s1_row["addr_digits"]:
        if len(d) >= 2 and d in index:
            candidates |= index[d]
    return candidates


def addr_token_block_candidates(s1_row, index):
    addr_tokens = set(s1_row["norm_address"].split()) - STOPWORDS - ADDR_STOPWORDS
    addr_tokens = {t for t in addr_tokens if len(t) >= 4}
    candidates = set()
    for tok in addr_tokens:
        candidates |= index.get(tok, set())
    return candidates


def build_faiss_indexes(df, embeddings):
    """country -> (faiss_index, entity_ids array). Requires `faiss` import by caller."""
    import faiss
    indexes = {}
    for country in df["country"].unique():
        mask = (df["country"] == country).values
        sub_emb = embeddings[mask].astype("float32")
        sub_ids = df["entity_id"].values[mask]
        index = faiss.IndexFlatIP(sub_emb.shape[1])
        index.add(sub_emb)
        indexes[country] = (index, sub_ids)
    return indexes


def faiss_block_candidates(s1_country, s1_vec, faiss_by_country, top_k=15):
    if s1_country not in faiss_by_country:
        return set()
    index, ids = faiss_by_country[s1_country]
    k = min(top_k, index.ntotal)
    if k == 0:
        return set()
    _, idxs = index.search(s1_vec.reshape(1, -1).astype("float32"), k)
    return set(ids[idxs[0]])


def quick_rank_score(s1_row, cand_id, s2_lookup, s3_lookup):
    """Cheap scoring used ONLY to rank/cap raw blocking candidates to top-K."""
    src = s2_lookup if cand_id.startswith("S2-") else s3_lookup
    if cand_id not in src:
        return 0.0
    cand_name, cand_addr = src[cand_id]
    name_overlap = len(set(s1_row["norm_name"].split()) & set(cand_name.split()))
    addr_overlap = len(set(s1_row["norm_address"].split()) & set(cand_addr.split()))
    digit_overlap = len(s1_row["addr_digits"] & set(re.findall(r"\d+", cand_addr)))
    return name_overlap * 2 + addr_overlap + digit_overlap * 3


def cap_candidates(candidate_map, s1_df, s2_lookup, s3_lookup, top_k=75):
    """Rank raw blocking union per S1 entity and keep only the top_k candidates."""
    capped = {}
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        raw = candidate_map.get(s1_id, set())
        if len(raw) <= top_k:
            capped[s1_id] = raw
        else:
            scored = [(c, quick_rank_score(s1_row, c, s2_lookup, s3_lookup)) for c in raw]
            scored.sort(key=lambda x: -x[1])
            capped[s1_id] = {c for c, _ in scored[:top_k]}
    return capped


# ============================================================
# STAGE 3: PAIRWISE FEATURE ENGINEERING
# ============================================================

FEATURE_COLS = [
    "name_jaccard", "addr_jaccard", "name_levenshtein", "addr_levenshtein",
    "name_exact", "name_len_diff", "addr_len_diff", "digit_jaccard",
    "digit_shared_count", "country_match", "embedding_cosine",
    "both_have_addr", "either_missing_addr", "addr_high_name_low",
]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def digit_overlap_features(digits_a, digits_b):
    if not digits_a and not digits_b:
        return 1.0, 0
    if not digits_a or not digits_b:
        return 0.0, 0
    shared = digits_a & digits_b
    union = digits_a | digits_b
    return len(shared) / len(union), len(shared)


def has_address(addr_text):
    return len(addr_text.strip()) > 0


def build_lookup(df):
    """entity_id -> dict of precomputed fields, for O(1) access."""
    return {
        row["entity_id"]: {
            "norm_name": row["norm_name"],
            "norm_address": row["norm_address"],
            "addr_digits": row["addr_digits"],
            "country": row["country"],
            "name_tokens": frozenset(row["norm_name"].split()),
            "addr_tokens": frozenset(row["norm_address"].split()),
        }
        for _, row in df.iterrows()
    }


def compute_features_batch(pairs_df, s1_lookup, s2_lookup, s3_lookup,
                            get_embedding_fn):
    """get_embedding_fn(entity_id) -> np.array embedding vector."""
    records = []
    for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        s1_info = s1_lookup[s1_id]
        c_info = s2_lookup.get(cand_id) or s3_lookup.get(cand_id)
        if c_info is None:
            continue

        name_jac = jaccard(s1_info["name_tokens"], c_info["name_tokens"])
        addr_jac = jaccard(s1_info["addr_tokens"], c_info["addr_tokens"])
        name_lev = fuzz.ratio(s1_info["norm_name"], c_info["norm_name"]) / 100.0
        addr_lev = fuzz.ratio(s1_info["norm_address"], c_info["norm_address"]) / 100.0
        name_exact = float(s1_info["norm_name"] == c_info["norm_name"])
        name_len_diff = abs(len(s1_info["norm_name"]) - len(c_info["norm_name"]))
        addr_len_diff = abs(len(s1_info["norm_address"]) - len(c_info["norm_address"]))
        digit_jac, digit_shared_count = digit_overlap_features(s1_info["addr_digits"], c_info["addr_digits"])
        country_match = float(s1_info["country"] == c_info["country"])

        s1_vec = get_embedding_fn(s1_id)
        c_vec = get_embedding_fn(cand_id)
        emb_cosine = float(np.dot(s1_vec, c_vec))

        s1_has_addr = has_address(s1_info["norm_address"])
        c_has_addr = has_address(c_info["norm_address"])
        both_have_addr = float(s1_has_addr and c_has_addr)
        either_missing_addr = float(not s1_has_addr or not c_has_addr)
        addr_high_name_low = float(addr_jac > 0.7 and name_jac < 0.15)

        records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "name_jaccard": name_jac,
            "addr_jaccard": addr_jac,
            "name_levenshtein": name_lev,
            "addr_levenshtein": addr_lev,
            "name_exact": name_exact,
            "name_len_diff": name_len_diff,
            "addr_len_diff": addr_len_diff,
            "digit_jaccard": digit_jac,
            "digit_shared_count": digit_shared_count,
            "country_match": country_match,
            "embedding_cosine": emb_cosine,
            "both_have_addr": both_have_addr,
            "either_missing_addr": either_missing_addr,
            "addr_high_name_low": addr_high_name_low,
        })
    return pd.DataFrame(records)


def flatten_candidates(candidate_map):
    rows = [(s1_id, c) for s1_id, cands in candidate_map.items() for c in cands]
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def build_labels(features_df, gt_map):
    pos_pairs = {(s1, tid) for s1, tids in gt_map.items() for tid in tids}
    features_df = features_df.copy()
    features_df["label"] = features_df.apply(
        lambda r: 1 if (r["source1_entity_id"], r["candidate_entity_id"]) in pos_pairs else 0, axis=1
    )
    return features_df


# ============================================================
# STAGE 5: SCORING (F_0.5)
# ============================================================

def f_beta_per_entity(predicted_ids, true_ids, beta=0.5):
    predicted_ids, true_ids = set(predicted_ids), set(true_ids)
    if not true_ids and not predicted_ids:
        return 1.0
    if not predicted_ids:
        return 0.0
    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids) if predicted_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta ** 2
    denom = (beta2 * precision) + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def score_submission(pred_df, gt_df):
    gt_map = dict(zip(gt_df["source1_entity_id"],
                       gt_df["matched_entity_ids"].fillna("").apply(
                           lambda x: x.split(",") if x.strip() else [])))
    pred_map = dict(zip(pred_df["source1_entity_id"],
                         pred_df["matched_entity_ids"].fillna("").apply(
                             lambda x: x.split(",") if x.strip() else [])))
    scores = [f_beta_per_entity(pred_map.get(s1, []), true_ids) for s1, true_ids in gt_map.items()]
    return sum(scores) / len(scores) if scores else 0.0


def build_predictions_at_threshold(pred_df, threshold, all_s1_ids):
    """pred_df needs columns: source1_entity_id, candidate_entity_id, pred_proba."""
    preds = defaultdict(set)
    for s1_id, cand_id, proba in zip(pred_df["source1_entity_id"],
                                       pred_df["candidate_entity_id"],
                                       pred_df["pred_proba"]):
        if proba >= threshold:
            preds[s1_id].add(cand_id)
    for s1_id in all_s1_ids:
        if s1_id not in preds:
            preds[s1_id] = set()
    return preds


print("pipeline.py loaded OK")
