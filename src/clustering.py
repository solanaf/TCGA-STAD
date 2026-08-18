#!/usr/bin/env python3
"""Common K-means clustering analysis for TCGA STAD embeddings.

This script applies the same K-means workflow to PCA, MDS, t-SNE, and UMAP
embeddings while keeping dimensionality reduction and clustering separate.

Primary vs sensitivity configurations
-------------------------------------
PCA and MDS currently contribute one canonical two-dimensional embedding each.
t-SNE and UMAP can contribute many cached sensitivity embeddings. Every cached
configuration is clustered independently across the same k grid, and the best
k for that configuration is selected by maximum silhouette score (ties favor
smaller k).

The canonical configuration from each method remains explicitly marked with
``is_primary=True``. This preserves the project's primary analysis while also
caching all configuration-level cluster solutions for later exploratory
comparison or Streamlit selection.

No clinical or technical metadata are used to select a dimensionality-reduction
configuration or k.

Expected project layout
-----------------------
processed/
    sample_info.tsv
analysis/
    pca/pca_scores.tsv.gz
    mds/mds_scores.tsv.gz
    tsne/tsne_scores.tsv.gz
    tsne/tsne_sensitivity_scores.tsv.gz          # preferred when present
    umap/umap_scores.tsv.gz
    umap/umap_sensitivity_scores.tsv.gz          # preferred when present

Outputs
-------
analysis/clustering/
    embedding_configurations.tsv
    kmeans_diagnostics.tsv
    kmeans_assignments_long.tsv.gz
    best_k_summary_all.tsv
    best_cluster_assignments_all_long.tsv.gz
    best_cluster_assignments_all_with_metadata.tsv.gz

    # Primary-analysis compatibility outputs
    best_k_summary.tsv
    best_cluster_assignments.tsv.gz
    best_cluster_assignments_with_metadata.tsv.gz

    clustering_summary.json
    figures/
        kmeans_silhouette_by_k_primary.png
        pca_kmeans_best.png
        mds_kmeans_best.png
        tsne_kmeans_best.png
        umap_kmeans_best.png

Run directly with:

python clustering.py \
    --analysis-root analysis \
    --processed-dir processed \
    --output-dir analysis/clustering

By default, all cached t-SNE/UMAP sensitivity configurations are included.
Use ``--primary-only`` to cluster only canonical configurations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    relative_path: str
    x_col: str
    y_col: str
    output_prefix: str
    sensitivity_path: str | None = None


@dataclass
class EmbeddingRun:
    method: str
    method_key: str
    configuration_id: str
    configuration_label: str
    is_primary: bool
    x_col: str
    y_col: str
    coordinates: pd.DataFrame
    source_file: str
    parameters: dict[str, object] = field(default_factory=dict)


EMBEDDING_SPECS: dict[str, EmbeddingSpec] = {
    "pca": EmbeddingSpec(
        name="PCA",
        relative_path="pca/pca_scores.tsv.gz",
        x_col="PC1",
        y_col="PC2",
        output_prefix="pca",
    ),
    "mds": EmbeddingSpec(
        name="MDS",
        relative_path="mds/mds_scores.tsv.gz",
        x_col="MDS1",
        y_col="MDS2",
        output_prefix="mds",
    ),
    "tsne": EmbeddingSpec(
        name="t-SNE",
        relative_path="tsne/tsne_scores.tsv.gz",
        sensitivity_path="tsne/tsne_sensitivity_scores.tsv.gz",
        x_col="tSNE1",
        y_col="tSNE2",
        output_prefix="tsne",
    ),
    "umap": EmbeddingSpec(
        name="UMAP",
        relative_path="umap/umap_scores.tsv.gz",
        sensitivity_path="umap/umap_sensitivity_scores.tsv.gz",
        x_col="UMAP1",
        y_col="UMAP2",
        output_prefix="umap",
    ),
}

DEFAULT_METHODS = ("pca", "mds", "tsne", "umap")
PARAMETER_COLUMNS = (
    "metric",
    "perplexity",
    "n_neighbors",
    "min_dist",
    "pre_pca_components",
)


def _safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _float_tag(value: object) -> str:
    value = float(value)
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _configuration_id_from_row(method_key: str, row: pd.Series) -> str:
    if "configuration_id" in row.index and pd.notna(row["configuration_id"]):
        return str(row["configuration_id"])

    if method_key == "tsne":
        metric = _safe_name(row.get("metric", "euclidean"))
        perplexity = _float_tag(row.get("perplexity", 30))
        return f"tsne__metric-{metric}__perplexity-{perplexity}"

    if method_key == "umap":
        metric = _safe_name(row.get("metric", "euclidean"))
        nn = int(float(row.get("n_neighbors", 15)))
        min_dist = _float_tag(row.get("min_dist", 0.1))
        return f"umap__metric-{metric}__nn-{nn}__mindist-{min_dist}"

    return f"{method_key}__primary"


def _configuration_label(method_key: str, params: dict[str, object]) -> str:
    if method_key == "tsne":
        return (
            f"metric={params.get('metric', 'euclidean')}, "
            f"perplexity={float(params.get('perplexity', 30)):g}"
        )
    if method_key == "umap":
        return (
            f"metric={params.get('metric', 'euclidean')}, "
            f"n_neighbors={int(float(params.get('n_neighbors', 15)))}, "
            f"min_dist={float(params.get('min_dist', 0.1)):g}"
        )
    return "primary"


def _validate_coordinates(
    df: pd.DataFrame,
    *,
    path: Path,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    required = {"sample_id", x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    coords = df[["sample_id", x_col, y_col]].copy()
    coords["sample_id"] = coords["sample_id"].astype(str)
    if coords["sample_id"].duplicated().any():
        dupes = coords.loc[coords["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id values in {path}: {dupes}")
    if coords[[x_col, y_col]].isna().any().any():
        raise ValueError(f"NaN coordinates found in {path}")
    if not np.isfinite(coords[[x_col, y_col]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite coordinates found in {path}")
    return coords


def load_canonical_embedding(analysis_root: Path, spec: EmbeddingSpec) -> EmbeddingRun:
    """Load one canonical embedding."""

    path = analysis_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {spec.name} embedding: {path}\n"
            "Run the corresponding dimensionality-reduction script first."
        )

    raw = pd.read_csv(path, sep="\t")
    coords = _validate_coordinates(raw, path=path, x_col=spec.x_col, y_col=spec.y_col)
    return EmbeddingRun(
        method=spec.name,
        method_key=spec.output_prefix,
        configuration_id=f"{spec.output_prefix}__primary",
        configuration_label="primary",
        is_primary=True,
        x_col=spec.x_col,
        y_col=spec.y_col,
        coordinates=coords,
        source_file=str(path),
        parameters={},
    )


def load_sensitivity_embeddings(
    analysis_root: Path,
    spec: EmbeddingSpec,
) -> list[EmbeddingRun]:
    """Load long-format cached sensitivity embeddings for t-SNE or UMAP."""

    if spec.sensitivity_path is None:
        return []
    path = analysis_root / spec.sensitivity_path
    if not path.exists():
        return []

    df = pd.read_csv(path, sep="\t")
    required = {"sample_id", spec.x_col, spec.y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "configuration_id" not in df.columns:
        # Backward compatibility with the first t-SNE cache implementation.
        df["configuration_id"] = df.apply(
            lambda row: _configuration_id_from_row(spec.output_prefix, row), axis=1
        )

    runs: list[EmbeddingRun] = []
    for config_id, group in df.groupby("configuration_id", sort=False, dropna=False):
        first = group.iloc[0]
        params: dict[str, object] = {}
        for column in PARAMETER_COLUMNS:
            if column in group.columns and pd.notna(first[column]):
                value = first[column]
                if isinstance(value, np.generic):
                    value = value.item()
                params[column] = value

        is_primary = (
            _truthy(first.get("is_primary", False))
            if "is_primary" in group.columns
            else False
        )
        label = (
            str(first["configuration_label"])
            if "configuration_label" in group.columns
            and pd.notna(first["configuration_label"])
            else _configuration_label(spec.output_prefix, params)
        )
        coords = _validate_coordinates(
            group,
            path=path,
            x_col=spec.x_col,
            y_col=spec.y_col,
        )
        runs.append(
            EmbeddingRun(
                method=spec.name,
                method_key=spec.output_prefix,
                configuration_id=str(config_id),
                configuration_label=label,
                is_primary=is_primary,
                x_col=spec.x_col,
                y_col=spec.y_col,
                coordinates=coords,
                source_file=str(path),
                parameters=params,
            )
        )
    return runs


def load_method_runs(
    analysis_root: Path,
    spec: EmbeddingSpec,
    *,
    include_sensitivity: bool,
) -> list[EmbeddingRun]:
    """Load all available runs for one dimensionality-reduction method."""

    if include_sensitivity and spec.sensitivity_path is not None:
        runs = load_sensitivity_embeddings(analysis_root, spec)
        if runs:
            primary_runs = [run for run in runs if run.is_primary]
            if not primary_runs:
                canonical = load_canonical_embedding(analysis_root, spec)
                # Infer which cached config corresponds to the canonical coordinates by
                # known parameter defaults when possible, otherwise append canonical.
                runs.append(canonical)
            return runs

    return [load_canonical_embedding(analysis_root, spec)]


def load_sample_info(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "sample_info.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample metadata: {path}")
    sample_info = pd.read_csv(path, sep="\t")
    if "sample_id" not in sample_info.columns:
        raise ValueError(f"{path} must contain sample_id")
    sample_info["sample_id"] = sample_info["sample_id"].astype(str)
    if sample_info["sample_id"].duplicated().any():
        raise ValueError(f"sample_id must be unique in {path}")
    return sample_info


def _base_metadata(run: EmbeddingRun) -> dict[str, object]:
    row: dict[str, object] = {
        "method": run.method,
        "method_key": run.method_key,
        "configuration_id": run.configuration_id,
        "configuration_label": run.configuration_label,
        "is_primary": bool(run.is_primary),
        "source_file": run.source_file,
    }
    for column in PARAMETER_COLUMNS:
        row[column] = run.parameters.get(column, np.nan)
    return row


def fit_kmeans_grid(
    run: EmbeddingRun,
    *,
    k_values: Iterable[int],
    n_init: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit K-means across k for one embedding configuration."""

    embedding = run.coordinates
    X = embedding[[run.x_col, run.y_col]].to_numpy(dtype=float)
    n_samples = X.shape[0]
    diagnostic_rows: list[dict[str, object]] = []
    assignment_rows: list[pd.DataFrame] = []
    base = _base_metadata(run)

    for k in k_values:
        if k < 2:
            raise ValueError("All k values must be >= 2")
        if k >= n_samples:
            raise ValueError(f"k={k} must be smaller than n_samples={n_samples}")

        model = KMeans(
            n_clusters=int(k),
            n_init=int(n_init),
            random_state=int(random_state),
        )
        labels_zero = model.fit_predict(X)
        labels = labels_zero + 1
        counts = pd.Series(labels).value_counts().sort_index()

        diagnostic_rows.append(
            {
                **base,
                "k": int(k),
                "silhouette": float(silhouette_score(X, labels_zero)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels_zero)),
                "davies_bouldin": float(davies_bouldin_score(X, labels_zero)),
                "inertia": float(model.inertia_),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "cluster_size_ratio": float(counts.min() / counts.max()),
            }
        )

        assignments = pd.DataFrame(
            {
                "sample_id": embedding["sample_id"].to_numpy(),
                **{key: value for key, value in base.items() if key != "source_file"},
                "k": int(k),
                "cluster": labels.astype(int),
            }
        )
        assignment_rows.append(assignments)

    return pd.DataFrame(diagnostic_rows), pd.concat(assignment_rows, ignore_index=True)


def select_best_k(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Select best k independently for each embedding configuration."""

    ordered = diagnostics.sort_values(
        ["configuration_id", "silhouette", "k"],
        ascending=[True, False, True],
    )
    best = (
        ordered.groupby("configuration_id", as_index=False, sort=False)
        .head(1)
        .sort_values(["method_key", "is_primary", "configuration_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return best


def build_selected_assignments_long(
    all_assignments: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return selected best-k labels for every configuration in long format."""

    keys = best_summary[["configuration_id", "k"]].drop_duplicates()
    selected = all_assignments.merge(
        keys,
        on=["configuration_id", "k"],
        how="inner",
        validate="many_to_one",
    )
    return selected.sort_values(["method_key", "configuration_id", "sample_id"]).reset_index(drop=True)


def build_primary_assignment_table(
    selected_long: pd.DataFrame,
    primary_best: pd.DataFrame,
) -> pd.DataFrame:
    """Build legacy wide one-cluster-column-per-method table for primary configs."""

    merged: pd.DataFrame | None = None
    for row in primary_best.itertuples(index=False):
        subset = selected_long.loc[
            selected_long["configuration_id"].eq(row.configuration_id),
            ["sample_id", "cluster"],
        ].copy()
        subset = subset.rename(columns={"cluster": f"{row.method_key}_cluster"})
        merged = subset if merged is None else merged.merge(
            subset, on="sample_id", how="outer", validate="one_to_one"
        )

    if merged is None:
        raise ValueError("No primary cluster assignments were generated")
    return merged


def configuration_manifest(runs: list[EmbeddingRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(_base_metadata(run))
    return pd.DataFrame(rows).sort_values(
        ["method_key", "is_primary", "configuration_id"],
        ascending=[True, False, True],
    )


def plot_silhouette_by_k(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Plot silhouette score versus k for primary configurations."""

    if diagnostics.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, group in diagnostics.groupby("method", sort=False):
        group = group.sort_values("k")
        ax.plot(group["k"], group["silhouette"], marker="o", label=method)

    ax.set_xlabel("Number of K-means clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means silhouette score across primary embeddings")
    ax.set_xticks(sorted(diagnostics["k"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_clusters(
    run: EmbeddingRun,
    labels: pd.DataFrame,
    *,
    k: int,
    silhouette: float,
    output_path: Path,
) -> None:
    """Plot selected K-means partition for one primary embedding."""

    data = run.coordinates.merge(
        labels[["sample_id", "cluster"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for cluster_id in sorted(data["cluster"].dropna().unique()):
        mask = data["cluster"] == cluster_id
        ax.scatter(
            data.loc[mask, run.x_col],
            data.loc[mask, run.y_col],
            s=28,
            alpha=0.85,
            label=f"Cluster {int(cluster_id)}",
        )

    ax.set_xlabel(run.x_col)
    ax.set_ylabel(run.y_col)
    ax.set_title(f"{run.method}: primary K-means (k={k}, silhouette={silhouette:.3f})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_terminal_summary(best_all: pd.DataFrame) -> str:
    primary = best_all.loc[best_all["is_primary"].map(_truthy)].copy()
    lines = [
        "TCGA STAD common K-means clustering summary",
        "-------------------------------------------",
        "Best k selected independently for each cached embedding configuration.",
        "Clinical/technical metadata were NOT used for configuration or k selection.",
        "",
        f"Configurations clustered: {len(best_all)}",
        f"Primary configurations:   {len(primary)}",
        f"Sensitivity configs:      {len(best_all) - len(primary)}",
        "",
        "Primary results:",
        f"{'Method':<8} {'Best k':>7} {'Silhouette':>12} {'CH score':>12} {'DB score':>10} {'Min/Max':>11}",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.method:<8} {int(row.k):>7d} {row.silhouette:>12.4f} "
            f"{row.calinski_harabasz:>12.2f} {row.davies_bouldin:>10.4f} "
            f"{int(row.min_cluster_size):>4d}/{int(row.max_cluster_size):<4d}"
        )
    return "\n".join(lines)


def run_clustering(
    *,
    analysis_root: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    methods: Iterable[str],
    k_min: int,
    k_max: int,
    n_init: int,
    random_state: int,
    include_sensitivity: bool,
) -> dict[str, object]:
    analysis_root = Path(analysis_root)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    method_keys = list(dict.fromkeys(methods))
    unknown = [m for m in method_keys if m not in EMBEDDING_SPECS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Choose from {sorted(EMBEDDING_SPECS)}")
    if not method_keys:
        raise ValueError("At least one method is required")
    if k_min < 2 or k_max < k_min:
        raise ValueError("Require 2 <= k_min <= k_max")
    if n_init < 1:
        raise ValueError("n_init must be >= 1")

    k_values = list(range(int(k_min), int(k_max) + 1))
    runs: list[EmbeddingRun] = []
    for key in method_keys:
        method_runs = load_method_runs(
            analysis_root,
            EMBEDDING_SPECS[key],
            include_sensitivity=include_sensitivity,
        )
        runs.extend(method_runs)

    # Ensure every configuration has the same sample set.
    reference_samples: set[str] | None = None
    for run in runs:
        sample_set = set(run.coordinates["sample_id"].astype(str))
        if reference_samples is None:
            reference_samples = sample_set
        elif sample_set != reference_samples:
            missing = sorted(reference_samples - sample_set)[:5]
            extra = sorted(sample_set - reference_samples)[:5]
            raise ValueError(
                f"Sample IDs do not match for {run.configuration_id}. "
                f"Missing examples={missing}; extra examples={extra}"
            )

    manifest = configuration_manifest(runs)
    manifest.to_csv(output_dir / "embedding_configurations.tsv", sep="\t", index=False)

    diagnostics_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    for run in runs:
        print(f"Clustering {run.method}: {run.configuration_label}")
        diagnostics, assignments = fit_kmeans_grid(
            run,
            k_values=k_values,
            n_init=n_init,
            random_state=random_state,
        )
        diagnostics_parts.append(diagnostics)
        assignment_parts.append(assignments)

    diagnostics_all = pd.concat(diagnostics_parts, ignore_index=True)
    assignments_all = pd.concat(assignment_parts, ignore_index=True)
    best_all = select_best_k(diagnostics_all)
    selected_all = build_selected_assignments_long(assignments_all, best_all)

    diagnostics_all.to_csv(output_dir / "kmeans_diagnostics.tsv", sep="\t", index=False)
    assignments_all.to_csv(
        output_dir / "kmeans_assignments_long.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    best_all.to_csv(output_dir / "best_k_summary_all.tsv", sep="\t", index=False)
    selected_all.to_csv(
        output_dir / "best_cluster_assignments_all_long.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    sample_info = load_sample_info(processed_dir)
    selected_all_with_metadata = selected_all.merge(
        sample_info,
        on="sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    selected_all_with_metadata.to_csv(
        output_dir / "best_cluster_assignments_all_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    # Primary-analysis compatibility outputs.
    primary_best = best_all.loc[best_all["is_primary"].map(_truthy)].copy()
    if primary_best["method_key"].duplicated().any():
        duplicates = primary_best.loc[
            primary_best["method_key"].duplicated(keep=False),
            ["method_key", "configuration_id"],
        ]
        raise ValueError(
            "More than one configuration is marked primary for a method:\n"
            + duplicates.to_string(index=False)
        )
    primary_best = primary_best.sort_values("method_key").reset_index(drop=True)
    primary_best.to_csv(output_dir / "best_k_summary.tsv", sep="\t", index=False)

    primary_assignments = build_primary_assignment_table(selected_all, primary_best)
    primary_assignments.to_csv(
        output_dir / "best_cluster_assignments.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    primary_with_metadata = primary_assignments.merge(
        sample_info,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    primary_with_metadata.to_csv(
        output_dir / "best_cluster_assignments_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    primary_diagnostics = diagnostics_all.loc[diagnostics_all["is_primary"].map(_truthy)]
    plot_silhouette_by_k(
        primary_diagnostics,
        figure_dir / "kmeans_silhouette_by_k_primary.png",
    )
    # Keep legacy figure filename too.
    plot_silhouette_by_k(
        primary_diagnostics,
        figure_dir / "kmeans_silhouette_by_k.png",
    )

    run_by_id = {run.configuration_id: run for run in runs}
    for row in primary_best.itertuples(index=False):
        run = run_by_id[str(row.configuration_id)]
        labels = selected_all.loc[
            selected_all["configuration_id"].eq(row.configuration_id),
            ["sample_id", "cluster"],
        ]
        plot_best_clusters(
            run,
            labels,
            k=int(row.k),
            silhouette=float(row.silhouette),
            output_path=figure_dir / f"{row.method_key}_kmeans_best.png",
        )

    per_method_counts = best_all.groupby("method_key").size().to_dict()
    summary: dict[str, object] = {
        "clustering": {
            "algorithm": "KMeans",
            "k_values": k_values,
            "selection_rule": "maximum silhouette within each embedding configuration; ties favor smaller k",
            "n_init": int(n_init),
            "random_state": int(random_state),
            "input_dimensions": 2,
            "metadata_used_for_selection": False,
            "sensitivity_configurations_included": bool(include_sensitivity),
            "n_configurations": int(len(best_all)),
        },
        "configuration_counts": {key: int(value) for key, value in per_method_counts.items()},
        "primary_methods": {
            row.method_key: {
                "name": row.method,
                "configuration_id": row.configuration_id,
                "configuration_label": row.configuration_label,
                "best_k": int(row.k),
                "silhouette": float(row.silhouette),
                "calinski_harabasz": float(row.calinski_harabasz),
                "davies_bouldin": float(row.davies_bouldin),
                "inertia": float(row.inertia),
                "min_cluster_size": int(row.min_cluster_size),
                "max_cluster_size": int(row.max_cluster_size),
            }
            for row in primary_best.itertuples(index=False)
        },
        "outputs": {
            "configuration_manifest": "embedding_configurations.tsv",
            "all_best_k": "best_k_summary_all.tsv",
            "all_selected_assignments": "best_cluster_assignments_all_long.tsv.gz",
            "all_selected_assignments_with_metadata": "best_cluster_assignments_all_with_metadata.tsv.gz",
            "primary_best_k": "best_k_summary.tsv",
            "primary_assignments": "best_cluster_assignments.tsv.gz",
        },
    }
    with (output_dir / "clustering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(format_terminal_summary(best_all))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one common K-means workflow to canonical and cached sensitivity "
            "PCA/MDS/t-SNE/UMAP embeddings."
        )
    )
    parser.add_argument(
        "--analysis-root",
        default="analysis",
        help="Root containing pca/, mds/, tsne/, and umap/ outputs (default: analysis)",
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Directory containing sample_info.tsv (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/clustering",
        help="Output directory (default: analysis/clustering)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(EMBEDDING_SPECS),
        default=list(DEFAULT_METHODS),
        help="Embeddings to cluster (default: pca mds tsne umap)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of K-means clusters (default: 2)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum number of K-means clusters (default: 10)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=50,
        help="K-means initializations for each k (default: 50)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Ignore cached t-SNE/UMAP sensitivity embeddings and cluster only primary configurations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_clustering(
        analysis_root=args.analysis_root,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        k_min=args.k_min,
        k_max=args.k_max,
        n_init=args.n_init,
        random_state=args.random_state,
        include_sensitivity=not args.primary_only,
    )


if __name__ == "__main__":
    main()