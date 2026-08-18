from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def read_tsv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(Path(path_string), sep="\t", low_memory=False)


@st.cache_data(show_spinner=False)
def read_json(path_string: str) -> dict[str, Any]:
    with Path(path_string).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def existing(root: Path, relative: str) -> Path | None:
    path = root / relative
    return path if path.exists() else None


def maybe_tsv(root: Path, relative: str) -> pd.DataFrame | None:
    path = existing(root, relative)
    return None if path is None else read_tsv(str(path))


def maybe_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = existing(root, relative)
    return None if path is None else read_json(str(path))


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def sample_info(root: Path) -> pd.DataFrame | None:
    df = maybe_tsv(root, "processed/sample_info.tsv")
    if df is not None and "sample_id" in df:
        df["sample_id"] = df["sample_id"].astype(str)
    return df


def merge_metadata(coords: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None or "sample_id" not in coords:
        return coords
    out = coords.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    md = metadata.copy()
    md["sample_id"] = md["sample_id"].astype(str)
    extra = [c for c in md.columns if c == "sample_id" or c not in out.columns]
    return out.merge(md[extra], on="sample_id", how="left", validate="many_to_one", sort=False)


def _tsne_config_id(metric: object, perplexity: object) -> str:
    p = f"{float(perplexity):g}".replace(".", "p").replace("-", "m")
    metric_text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(metric))
    return f"tsne__metric-{metric_text}__perplexity-{p}"


def embedding_coordinates(root: Path, method_key: str, configuration_id: str | None = None) -> pd.DataFrame | None:
    method_key = method_key.lower()
    if method_key == "pca":
        return maybe_tsv(root, "analysis/pca/pca_scores.tsv.gz")
    if method_key == "mds":
        return maybe_tsv(root, "analysis/mds/mds_scores.tsv.gz")

    if method_key == "tsne":
        df = maybe_tsv(root, "analysis/tsne/tsne_sensitivity_scores.tsv.gz")
        if df is None:
            return maybe_tsv(root, "analysis/tsne/tsne_scores.tsv.gz")
        if "configuration_id" not in df.columns:
            df = df.copy()
            df["configuration_id"] = [
                _tsne_config_id(metric, p) for metric, p in zip(df["metric"], df["perplexity"])
            ]
        if configuration_id is None:
            return df
        return df.loc[df["configuration_id"].astype(str).eq(str(configuration_id))].copy()

    if method_key == "umap":
        df = maybe_tsv(root, "analysis/umap/umap_sensitivity_scores.tsv.gz")
        if df is None:
            return maybe_tsv(root, "analysis/umap/umap_scores.tsv.gz")
        if configuration_id is None:
            return df
        if "configuration_id" in df.columns:
            return df.loc[df["configuration_id"].astype(str).eq(str(configuration_id))].copy()
        return df

    raise KeyError(method_key)


def configuration_manifest(root: Path) -> pd.DataFrame | None:
    manifest = maybe_tsv(root, "analysis/clustering/embedding_configurations.tsv")
    if manifest is not None:
        return manifest

    # Fallback manifest if clustering has not been rerun yet.
    rows: list[dict[str, object]] = []
    for key, name in (("pca", "PCA"), ("mds", "MDS")):
        coords = embedding_coordinates(root, key)
        if coords is not None:
            rows.append({
                "method": name,
                "method_key": key,
                "configuration_id": f"{key}__primary",
                "configuration_label": "primary",
                "is_primary": True,
            })
    tsne = embedding_coordinates(root, "tsne")
    if tsne is not None and "configuration_id" in tsne:
        cols = [c for c in ["configuration_id", "metric", "perplexity"] if c in tsne]
        for _, row in tsne[cols].drop_duplicates().iterrows():
            rows.append({
                "method": "t-SNE", "method_key": "tsne",
                "configuration_id": row["configuration_id"],
                "configuration_label": f"metric={row.get('metric')}, perplexity={row.get('perplexity'):g}",
                "is_primary": str(row.get("metric")) == "euclidean" and float(row.get("perplexity")) == 30.0,
                "metric": row.get("metric"), "perplexity": row.get("perplexity"),
            })
    umap = embedding_coordinates(root, "umap")
    if umap is not None and "configuration_id" in umap:
        cols = [c for c in ["configuration_id", "configuration_label", "is_primary", "metric", "n_neighbors", "min_dist"] if c in umap]
        for _, row in umap[cols].drop_duplicates().iterrows():
            rows.append({"method": "UMAP", "method_key": "umap", **row.to_dict()})
    return pd.DataFrame(rows) if rows else None


def safe_numeric(value: object, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def coord_columns(method_key: str, df: pd.DataFrame) -> tuple[str, str]:
    expected = {
        "pca": ("PC1", "PC2"),
        "mds": ("MDS1", "MDS2"),
        "tsne": ("tSNE1", "tSNE2"),
        "umap": ("UMAP1", "UMAP2"),
    }[method_key]
    if all(c in df.columns for c in expected):
        return expected
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in {"cluster", "k"}]
    if len(numeric) < 2:
        raise ValueError(f"Could not identify two coordinate columns for {method_key}")
    return numeric[0], numeric[1]
