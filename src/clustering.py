#!/usr/bin/env python3
"""Common K-means clustering analysis for TCGA STAD embeddings.

This script is the first downstream analysis shared by PCA, MDS, t-SNE, and
UMAP. It deliberately separates *dimensionality reduction* from *clustering*:
each method first produces its canonical embedding, then this script applies
the same K-means procedure to the two-dimensional coordinates from every
method.

The number of clusters is not chosen using clinical metadata. Instead, K-means
is fit across a user-defined range of k values (default 2-10), and the primary
k for each embedding is selected by the highest silhouette score. Calinski-
Harabasz score, Davies-Bouldin score, inertia, and cluster-size diagnostics are
also retained so the choice can be inspected rather than treated as a black
box.

Important interpretation note
-----------------------------
These are *clusters in the 2-D embeddings*. t-SNE and UMAP intentionally reshape
local geometry, so a high silhouette score does not by itself prove that a
cluster is biologically meaningful. Biological/technical interpretation is a
separate downstream metadata-association step. The point here is to define a
single, reproducible clustering procedure that can be applied consistently to
all four embeddings.

Expected project layout
-----------------------
processed/
    sample_info.tsv
analysis/
    pca/pca_scores.tsv.gz
    mds/mds_scores.tsv.gz
    tsne/tsne_scores.tsv.gz
    umap/umap_scores.tsv.gz

Outputs
-------
analysis/clustering/
    kmeans_diagnostics.tsv
    kmeans_assignments_long.tsv.gz
    best_k_summary.tsv
    best_cluster_assignments.tsv.gz
    best_cluster_assignments_with_metadata.tsv.gz
    clustering_summary.json
    figures/
        kmeans_silhouette_by_k.png
        pca_kmeans_best.png
        mds_kmeans_best.png
        tsne_kmeans_best.png
        umap_kmeans_best.png

Run directly with:
python clustering.py \
    --analysis-root analysis \
    --processed-dir processed \
    --output-dir analysis/clustering

Assuming file structure is like:
src/
├── processed/
└── analysis/
    ├── pca/
    ├── mds/
    ├── tsne/
    └── umap/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
        x_col="tSNE1",
        y_col="tSNE2",
        output_prefix="tsne",
    ),
    "umap": EmbeddingSpec(
        name="UMAP",
        relative_path="umap/umap_scores.tsv.gz",
        x_col="UMAP1",
        y_col="UMAP2",
        output_prefix="umap",
    ),
}

DEFAULT_METHODS = ("pca", "mds", "tsne", "umap")


def load_embedding(analysis_root: Path, spec: EmbeddingSpec) -> pd.DataFrame:
    """Load one canonical embedding and validate sample/coordinate columns."""

    path = analysis_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {spec.name} embedding: {path}\n"
            "Run the corresponding dimensionality-reduction script first."
        )

    df = pd.read_csv(path, sep="\t")
    required = {"sample_id", spec.x_col, spec.y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id values in {path}: {dupes}")

    coords = df[["sample_id", spec.x_col, spec.y_col]].copy()
    if coords[[spec.x_col, spec.y_col]].isna().any().any():
        raise ValueError(f"NaN coordinates found in {path}")
    if not np.isfinite(coords[[spec.x_col, spec.y_col]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite coordinates found in {path}")
    return coords


def load_sample_info(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "sample_info.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample metadata: {path}")
    sample_info = pd.read_csv(path, sep="\t")
    if "sample_id" not in sample_info.columns:
        raise ValueError(f"{path} must contain sample_id")
    if sample_info["sample_id"].duplicated().any():
        raise ValueError(f"sample_id must be unique in {path}")
    return sample_info


def fit_kmeans_grid(
    embedding: pd.DataFrame,
    spec: EmbeddingSpec,
    *,
    k_values: Iterable[int],
    n_init: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit K-means across k and return diagnostics plus long assignments."""

    X = embedding[[spec.x_col, spec.y_col]].to_numpy(dtype=float)
    n_samples = X.shape[0]
    diagnostic_rows: list[dict[str, object]] = []
    assignment_rows: list[pd.DataFrame] = []

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
        labels = labels_zero + 1  # human-friendly 1-based cluster labels

        counts = pd.Series(labels).value_counts().sort_index()
        diagnostic_rows.append(
            {
                "method": spec.name,
                "method_key": spec.output_prefix,
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

        assignment_rows.append(
            pd.DataFrame(
                {
                    "sample_id": embedding["sample_id"].to_numpy(),
                    "method": spec.name,
                    "method_key": spec.output_prefix,
                    "k": int(k),
                    "cluster": labels.astype(int),
                }
            )
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True)
    return diagnostics, assignments


def select_best_k(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-silhouette k for each embedding; ties favor smaller k."""

    ordered = diagnostics.sort_values(
        ["method_key", "silhouette", "k"],
        ascending=[True, False, True],
    )
    best = ordered.groupby("method_key", as_index=False, sort=False).head(1).copy()
    best = best.sort_values("method_key").reset_index(drop=True)
    return best


def build_best_assignment_table(
    all_assignments: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per sample with the selected cluster from each method."""

    merged: pd.DataFrame | None = None
    for row in best_summary.itertuples(index=False):
        subset = all_assignments.loc[
            (all_assignments["method_key"] == row.method_key)
            & (all_assignments["k"] == row.k),
            ["sample_id", "cluster"],
        ].copy()
        subset = subset.rename(columns={"cluster": f"{row.method_key}_cluster"})
        merged = subset if merged is None else merged.merge(
            subset, on="sample_id", how="outer", validate="one_to_one"
        )

    if merged is None:
        raise ValueError("No best cluster assignments were generated")
    return merged


def plot_silhouette_by_k(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Plot silhouette score versus k for all dimensionality-reduction methods."""

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, group in diagnostics.groupby("method", sort=False):
        group = group.sort_values("k")
        ax.plot(group["k"], group["silhouette"], marker="o", label=method)

    ax.set_xlabel("Number of K-means clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means silhouette score across embeddings")
    ax.set_xticks(sorted(diagnostics["k"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_clusters(
    embedding: pd.DataFrame,
    spec: EmbeddingSpec,
    labels: pd.Series,
    *,
    k: int,
    silhouette: float,
    output_path: Path,
) -> None:
    """Plot the selected K-means partition on one embedding."""

    data = embedding.copy()
    data["cluster"] = labels.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    for cluster_id in sorted(data["cluster"].unique()):
        mask = data["cluster"] == cluster_id
        ax.scatter(
            data.loc[mask, spec.x_col],
            data.loc[mask, spec.y_col],
            s=28,
            alpha=0.85,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel(spec.x_col)
    ax.set_ylabel(spec.y_col)
    ax.set_title(
        f"{spec.name}: K-means clustering (k={k}, silhouette={silhouette:.3f})"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_terminal_summary(best: pd.DataFrame) -> str:
    lines = [
        "TCGA STAD common K-means clustering summary",
        "-------------------------------------------",
        "Best k selected independently for each embedding by maximum silhouette score.",
        "Clinical/technical metadata were NOT used to select k.",
        "",
        f"{'Method':<8} {'Best k':>7} {'Silhouette':>12} {'CH score':>12} {'DB score':>10} {'Min/Max':>11}",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"{row.method:<8} {row.k:>7d} {row.silhouette:>12.4f} "
            f"{row.calinski_harabasz:>12.2f} {row.davies_bouldin:>10.4f} "
            f"{row.min_cluster_size:>4d}/{row.max_cluster_size:<4d}"
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
        raise ValueError(
            f"Unknown methods: {unknown}. Choose from {sorted(EMBEDDING_SPECS)}"
        )
    if not method_keys:
        raise ValueError("At least one method is required")
    if k_min < 2 or k_max < k_min:
        raise ValueError("Require 2 <= k_min <= k_max")
    if n_init < 1:
        raise ValueError("n_init must be >= 1")

    k_values = list(range(int(k_min), int(k_max) + 1))
    embeddings: dict[str, pd.DataFrame] = {}
    diagnostics_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []

    reference_samples: set[str] | None = None
    for key in method_keys:
        spec = EMBEDDING_SPECS[key]
        embedding = load_embedding(analysis_root, spec)
        sample_set = set(embedding["sample_id"])
        if reference_samples is None:
            reference_samples = sample_set
        elif sample_set != reference_samples:
            missing = sorted(reference_samples - sample_set)[:5]
            extra = sorted(sample_set - reference_samples)[:5]
            raise ValueError(
                f"Sample IDs do not match across embeddings for {spec.name}. "
                f"Missing examples={missing}; extra examples={extra}"
            )

        diagnostics, assignments = fit_kmeans_grid(
            embedding,
            spec,
            k_values=k_values,
            n_init=n_init,
            random_state=random_state,
        )
        embeddings[key] = embedding
        diagnostics_parts.append(diagnostics)
        assignment_parts.append(assignments)

    diagnostics_all = pd.concat(diagnostics_parts, ignore_index=True)
    assignments_all = pd.concat(assignment_parts, ignore_index=True)
    best = select_best_k(diagnostics_all)

    diagnostics_all.to_csv(output_dir / "kmeans_diagnostics.tsv", sep="\t", index=False)
    assignments_all.to_csv(
        output_dir / "kmeans_assignments_long.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    best.to_csv(output_dir / "best_k_summary.tsv", sep="\t", index=False)

    best_assignments = build_best_assignment_table(assignments_all, best)
    best_assignments.to_csv(
        output_dir / "best_cluster_assignments.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    sample_info = load_sample_info(processed_dir)
    best_with_metadata = best_assignments.merge(
        sample_info,
        on="sample_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    best_with_metadata.to_csv(
        output_dir / "best_cluster_assignments_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    plot_silhouette_by_k(
        diagnostics_all,
        figure_dir / "kmeans_silhouette_by_k.png",
    )

    for row in best.itertuples(index=False):
        key = row.method_key
        spec = EMBEDDING_SPECS[key]
        chosen = assignments_all.loc[
            (assignments_all["method_key"] == key)
            & (assignments_all["k"] == row.k),
            ["sample_id", "cluster"],
        ]
        aligned = embeddings[key][["sample_id"]].merge(
            chosen,
            on="sample_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        plot_best_clusters(
            embeddings[key],
            spec,
            aligned["cluster"],
            k=int(row.k),
            silhouette=float(row.silhouette),
            output_path=figure_dir / f"{key}_kmeans_best.png",
        )

    summary: dict[str, object] = {
        "clustering": {
            "algorithm": "KMeans",
            "k_values": k_values,
            "selection_rule": "maximum silhouette; ties favor smaller k",
            "n_init": int(n_init),
            "random_state": int(random_state),
            "input_dimensions": 2,
            "metadata_used_for_selection": False,
        },
        "methods": {
            row.method_key: {
                "name": row.method,
                "best_k": int(row.k),
                "silhouette": float(row.silhouette),
                "calinski_harabasz": float(row.calinski_harabasz),
                "davies_bouldin": float(row.davies_bouldin),
                "inertia": float(row.inertia),
                "min_cluster_size": int(row.min_cluster_size),
                "max_cluster_size": int(row.max_cluster_size),
            }
            for row in best.itertuples(index=False)
        },
    }
    with (output_dir / "clustering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(format_terminal_summary(best))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply one common K-means clustering workflow to PCA/MDS/t-SNE/UMAP embeddings."
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
    )


if __name__ == "__main__":
    main()