#!/usr/bin/env python3
"""t-SNE analysis for the rebuilt TCGA STAD project.

This script consumes the canonical ``X_dimred`` matrix produced by
``preprocessing.py`` and computes a two-dimensional t-distributed stochastic
neighbor embedding (t-SNE). No additional gene filtering, scaling, clustering,
or metadata-driven parameter selection is performed.

The primary analysis uses a reproducible, modern scikit-learn baseline:

* 2 embedding dimensions
* perplexity = 30
* early exaggeration = 12
* learning rate = ``auto``
* initialization = PCA
* Euclidean distance
* Barnes-Hut optimization
* random_state = 42

Because t-SNE layouts can change substantially with perplexity and distance
metric, the script also supports a two-dimensional sensitivity sweep. By
default it evaluates perplexities 5, 15, 30, and 50 across Euclidean, cosine,
and correlation distances. The sweep is *not* optimized using K-means, DBSCAN,
or clinical metadata. Instead, each embedding is characterized using the t-SNE
optimization objective (KL divergence) and neighborhood-preservation metrics.
All sweep coordinates are cached to disk for later downstream comparison.

Neighborhood diagnostics
------------------------
``trustworthiness``
    Measures whether neighbors in the low-dimensional embedding were also
    neighbors in the original high-dimensional matrix. Values approach 1 when
    local structure is well preserved.

``mean Jaccard kNN overlap``
    For each sample, compares the set of its k nearest neighbors in the original
    matrix with the set in the t-SNE embedding, then averages the Jaccard index
    across samples. The default k values are 15 and 30, matching the local
    neighborhood scales emphasized in the project reference manuscript.

An optional PCA pre-reduction is available through ``--pre-pca-components``.
It is disabled by default so PCA, MDS, t-SNE, and UMAP can all consume the same
canonical matrix directly. It can be enabled later as a t-SNE sensitivity or
performance analysis.

Expected inputs in ``--processed-dir``
--------------------------------------
X_dimred.tsv.gz
    Samples x genes canonical dimensionality-reduction matrix.
sample_info.tsv
    Sample identifiers plus patient, technical, and clinical metadata.

Clustering and formal metadata-association testing are intentionally omitted.
Those analyses will be applied later using one common downstream workflow.

Run directly with:

python tsne.py \
    --processed-dir processed \
    --output-dir analysis/tsne \
    --color-by \
        center_code \
        gender \
        race \
        ajcc_pathologic_tumor_stage \
        histological_grade \
        vital_status


Previous version:

python tsne.py \
    --processed-dir processed \
    --output-dir analysis/tsne_historical \
    --perplexity 5 \
    --learning-rate 500 \
    --max-iter 2000 \
    --early-exaggeration 8 \
    --init random \
    --metric cosine \
    --perplexity-grid

Additional option to reduce dimensionality further beforehand: --pre-pca-components 50

Do sensitivity testing & assess -> KL divergence, trustworthiness @ 15, trustworthiness @ 30, kNN Jaccard @ 15, kNN Jaccard @ 30:

Sensitivity A — neighborhood scale
perplexity:
5, 15, 30, 50

Sensitivity B — distance definition
metric:
euclidean
cosine
correlation
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors


DEFAULT_COLOR_BY = ("center_code",)
DEFAULT_PERPLEXITY_GRID = (5.0, 15.0, 30.0, 50.0)
DEFAULT_METRIC_GRID = ("euclidean", "cosine", "correlation")
DEFAULT_NEIGHBORHOOD_K = (15, 30)


def load_inputs(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate canonical preprocessing outputs."""

    processed_dir = Path(processed_dir)
    x_path = processed_dir / "X_dimred.tsv.gz"
    sample_path = processed_dir / "sample_info.tsv"

    missing = [p.name for p in (x_path, sample_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required preprocessing output(s) in {processed_dir}: "
            + ", ".join(missing)
        )

    X = pd.read_csv(x_path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.index.name = "sample_id"

    sample_info = pd.read_csv(
        sample_path,
        sep="\t",
        dtype={
            "patient_id": "string",
            "sample_id": "string",
            "sample_type_code": "string",
            "tss_code": "string",
            "center_code": "string",
        },
    )

    if "sample_id" not in sample_info.columns:
        raise ValueError("sample_info.tsv is missing required column: sample_id")
    if sample_info["sample_id"].duplicated().any():
        raise ValueError("sample_info.tsv contains duplicate sample_id values")
    if X.index.duplicated().any():
        raise ValueError("X_dimred.tsv.gz contains duplicate sample_id values")

    x_ids = set(X.index)
    metadata_ids = set(sample_info["sample_id"].astype(str))
    missing_metadata = x_ids - metadata_ids
    extra_metadata = metadata_ids - x_ids
    if missing_metadata or extra_metadata:
        raise ValueError(
            "Sample mismatch between X_dimred and sample_info. "
            f"Missing metadata for {len(missing_metadata)} sample(s); "
            f"metadata has {len(extra_metadata)} extra sample(s)."
        )

    sample_info = sample_info.set_index("sample_id").loc[X.index].reset_index()

    values = X.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("X_dimred contains NaN or infinite values")

    return X, sample_info


def parse_learning_rate(value: str) -> str | float:
    """Accept either ``auto`` or a positive floating-point learning rate."""

    if value.lower() == "auto":
        return "auto"
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "learning rate must be 'auto' or a positive number"
        ) from exc
    if numeric <= 0:
        raise argparse.ArgumentTypeError("learning rate must be > 0")
    return numeric


def prepare_tsne_input(
    X: pd.DataFrame,
    pre_pca_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optionally reduce the canonical matrix with PCA before t-SNE."""

    values = X.to_numpy(dtype=float, copy=False)
    if pre_pca_components <= 0:
        return values, {
            "pre_pca_enabled": False,
            "pre_pca_components": None,
            "pre_pca_variance_percent": None,
        }

    max_components = min(X.shape[0] - 1, X.shape[1])
    if pre_pca_components > max_components:
        raise ValueError(
            f"--pre-pca-components={pre_pca_components} exceeds the maximum "
            f"informative rank ({max_components})"
        )

    pca = PCA(
        n_components=pre_pca_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    reduced = pca.fit_transform(values)
    return reduced, {
        "pre_pca_enabled": True,
        "pre_pca_components": int(pre_pca_components),
        "pre_pca_variance_percent": float(
            pca.explained_variance_ratio_.sum() * 100.0
        ),
    }


def _build_tsne_model(
    *,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: str | float,
    max_iter: int,
    init: str,
    metric: str,
    method: str,
    angle: float,
    n_jobs: int,
    random_state: int,
    verbose: int,
) -> TSNE:
    """Construct TSNE while tolerating older scikit-learn parameter names."""

    signature = inspect.signature(TSNE)
    params = signature.parameters

    kwargs: dict[str, object] = {
        "n_components": 2,
        "perplexity": perplexity,
        "early_exaggeration": early_exaggeration,
        "learning_rate": learning_rate,
        "init": init,
        "metric": metric,
        "method": method,
        "angle": angle,
        "random_state": random_state,
        "verbose": verbose,
    }

    if "max_iter" in params:
        kwargs["max_iter"] = max_iter
    elif "n_iter" in params:
        kwargs["n_iter"] = max_iter

    if "n_jobs" in params:
        kwargs["n_jobs"] = n_jobs

    return TSNE(**kwargs)


def fit_tsne(
    X_input: np.ndarray,
    *,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: str | float,
    max_iter: int,
    init: str,
    metric: str,
    method: str,
    angle: float,
    n_jobs: int,
    random_state: int,
    verbose: int = 0,
) -> tuple[TSNE, np.ndarray]:
    """Fit one two-dimensional t-SNE embedding."""

    n_samples = X_input.shape[0]
    if not 0 < perplexity < n_samples:
        raise ValueError(
            f"perplexity must be > 0 and < n_samples ({n_samples}); got {perplexity}"
        )
    if early_exaggeration <= 0:
        raise ValueError("early exaggeration must be > 0")
    if max_iter < 250:
        raise ValueError("t-SNE max_iter must be at least 250")
    if init not in {"pca", "random"}:
        raise ValueError("--init must be 'pca' or 'random'")
    if method not in {"barnes_hut", "exact"}:
        raise ValueError("--method must be 'barnes_hut' or 'exact'")

    model = _build_tsne_model(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init=init,
        metric=metric,
        method=method,
        angle=angle,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    embedding = model.fit_transform(X_input)
    if not np.isfinite(embedding).all():
        raise ValueError("t-SNE produced NaN or infinite coordinates")
    return model, embedding


def _knn_indices(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str,
    n_jobs: int,
) -> np.ndarray:
    """Return k nearest neighbors for each row, excluding the row itself."""

    if not 1 <= n_neighbors < X.shape[0]:
        raise ValueError(
            f"n_neighbors must be between 1 and n_samples-1; got {n_neighbors}"
        )

    # Ask for k+1 because the nearest point to each sample is itself.
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        n_jobs=n_jobs,
    )
    model.fit(X)
    indices = model.kneighbors(return_distance=False)

    rows: list[np.ndarray] = []
    for i, row in enumerate(indices):
        filtered = row[row != i]
        if len(filtered) < n_neighbors:
            # Extremely rare tied/self behavior: request a slightly larger pool.
            fallback = NearestNeighbors(
                n_neighbors=min(X.shape[0], n_neighbors + 2),
                metric=metric,
                n_jobs=n_jobs,
            ).fit(X)
            expanded = fallback.kneighbors(X[i : i + 1], return_distance=False)[0]
            filtered = expanded[expanded != i]
        rows.append(filtered[:n_neighbors])

    return np.vstack(rows)


def mean_jaccard_neighbor_overlap(
    X_original: np.ndarray,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    original_metric: str,
    n_jobs: int,
) -> float:
    """Mean Jaccard overlap of original-space and embedding-space kNN sets."""

    original_neighbors = _knn_indices(
        X_original,
        n_neighbors=n_neighbors,
        metric=original_metric,
        n_jobs=n_jobs,
    )
    embedded_neighbors = _knn_indices(
        embedding,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=n_jobs,
    )

    scores = []
    for original_row, embedded_row in zip(original_neighbors, embedded_neighbors):
        a = set(original_row.tolist())
        b = set(embedded_row.tolist())
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(scores))


def embedding_diagnostics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    model: TSNE,
    *,
    original_metric: str,
    neighborhood_k: Sequence[int],
    n_jobs: int,
) -> dict[str, float | int | None]:
    """Compute optimization and local-neighborhood diagnostics."""

    diagnostics: dict[str, float | int | None] = {
        "kl_divergence": float(model.kl_divergence_),
        "n_iter": int(getattr(model, "n_iter_", -1)),
        "effective_learning_rate": (
            float(getattr(model, "learning_rate_"))
            if hasattr(model, "learning_rate_")
            else None
        ),
    }

    trustworthiness_params = inspect.signature(trustworthiness).parameters
    for k in neighborhood_k:
        if k >= X_original.shape[0] / 2:
            raise ValueError(
                f"trustworthiness requires k < n_samples/2; got k={k}"
            )
        trust_kwargs: dict[str, object] = {"n_neighbors": k}
        if "metric" in trustworthiness_params:
            trust_kwargs["metric"] = original_metric
        diagnostics[f"trustworthiness_k{k}"] = float(
            trustworthiness(X_original, embedding, **trust_kwargs)
        )
        diagnostics[f"mean_jaccard_k{k}"] = mean_jaccard_neighbor_overlap(
            X_original,
            embedding,
            n_neighbors=k,
            original_metric=original_metric,
            n_jobs=n_jobs,
        )

    return diagnostics


def _is_numeric_metadata(series: pd.Series) -> bool:
    """Treat only actual numeric dtypes as continuous metadata."""

    return pd.api.types.is_numeric_dtype(series)


def plot_embedding(
    data: pd.DataFrame,
    output_path: Path,
    *,
    color_by: str | None = None,
    perplexity: float,
    metric: str,
) -> None:
    """Save a two-dimensional t-SNE scatterplot."""

    fig, ax = plt.subplots(figsize=(7, 6))

    if color_by is None:
        ax.scatter(data["tSNE1"], data["tSNE2"], s=28, alpha=0.8)
    else:
        if color_by not in data.columns:
            raise KeyError(f"Metadata column not found: {color_by}")

        series = data[color_by]
        valid = series.notna()

        if _is_numeric_metadata(series):
            numeric = pd.to_numeric(series, errors="coerce")
            scatter = ax.scatter(
                data.loc[valid, "tSNE1"],
                data.loc[valid, "tSNE2"],
                c=numeric.loc[valid],
                s=28,
                alpha=0.85,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_by)
        else:
            categories = sorted(series.loc[valid].astype(str).unique())
            for category in categories:
                mask = valid & series.astype("string").eq(category)
                ax.scatter(
                    data.loc[mask, "tSNE1"],
                    data.loc[mask, "tSNE2"],
                    s=28,
                    alpha=0.8,
                    label=category,
                )

        missing = ~valid
        if missing.any():
            ax.scatter(
                data.loc[missing, "tSNE1"],
                data.loc[missing, "tSNE2"],
                s=28,
                alpha=0.45,
                marker="x",
                label="Missing" if not _is_numeric_metadata(series) else None,
            )

        if not _is_numeric_metadata(series) and (valid.any() or missing.any()):
            ax.legend(
                title=color_by,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                fontsize=8,
            )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if color_by is None:
        title = f"t-SNE embedding ({metric}, perplexity={perplexity:g})"
    else:
        title = f"t-SNE — {color_by} ({metric}, perplexity={perplexity:g})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity_sweep(
    embeddings: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    """Save side-by-side uncolored embeddings across perplexity values."""

    perplexities = sorted(embeddings)
    if not perplexities:
        return

    n = len(perplexities)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, perplexity in zip(axes_array, perplexities):
        embedding = embeddings[perplexity]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=22, alpha=0.8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"Perplexity = {perplexity:g}")

    for ax in axes_array[len(perplexities) :]:
        ax.axis("off")

    fig.suptitle("t-SNE sensitivity to perplexity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_sweep(
    embeddings: dict[tuple[str, float], np.ndarray],
    output_path: Path,
    *,
    metrics: Sequence[str],
    perplexities: Sequence[float],
) -> None:
    """Save a metric-by-perplexity grid of uncolored t-SNE embeddings."""

    if not embeddings or not metrics or not perplexities:
        return

    nrows = len(metrics)
    ncols = len(perplexities)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    for row_idx, metric_name in enumerate(metrics):
        for col_idx, p in enumerate(perplexities):
            ax = axes[row_idx, col_idx]
            embedding = embeddings.get((metric_name, float(p)))
            if embedding is None:
                ax.axis("off")
                continue
            ax.scatter(embedding[:, 0], embedding[:, 1], s=18, alpha=0.8)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title(f"{metric_name} | perplexity={p:g}")

    fig.suptitle("t-SNE sensitivity to distance metric and perplexity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_sweep(
    embeddings: dict[str, np.ndarray],
    output_path: Path,
    *,
    perplexity: float,
) -> None:
    """Save side-by-side uncolored embeddings across distance metrics."""

    metrics = list(embeddings)
    if not metrics:
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.2), squeeze=False)
    axes_array = axes.ravel()

    for ax, metric_name in zip(axes_array, metrics):
        embedding = embeddings[metric_name]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=22, alpha=0.8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(metric_name)

    fig.suptitle(f"t-SNE sensitivity to distance metric (perplexity={perplexity:g})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity_diagnostics(
    table: pd.DataFrame,
    output_path: Path,
    neighborhood_k: Sequence[int],
) -> None:
    """Save local-preservation diagnostics across perplexity values."""

    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_table = table.sort_values("perplexity")

    for k in neighborhood_k:
        trust_col = f"trustworthiness_k{k}"
        if trust_col in sorted_table.columns:
            ax.plot(
                sorted_table["perplexity"],
                sorted_table[trust_col],
                marker="o",
                label=f"Trustworthiness (k={k})",
            )
        jaccard_col = f"mean_jaccard_k{k}"
        if jaccard_col in sorted_table.columns:
            ax.plot(
                sorted_table["perplexity"],
                sorted_table[jaccard_col],
                marker="o",
                linestyle="--",
                label=f"Jaccard kNN overlap (k={k})",
            )

    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Neighborhood preservation")
    ax.set_ylim(0, 1.02)
    ax.set_title("t-SNE neighborhood preservation by perplexity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def run_analysis(
    X: pd.DataFrame,
    sample_info: pd.DataFrame,
    *,
    output_dir: str | Path,
    perplexity: float,
    perplexity_grid: Iterable[float],
    early_exaggeration: float,
    learning_rate: str | float,
    max_iter: int,
    init: str,
    metric: str,
    metric_grid: Iterable[str],
    method: str,
    angle: float,
    n_jobs: int,
    random_state: int,
    pre_pca_components: int,
    neighborhood_k: Sequence[int],
    color_by: Iterable[str],
    verbose: int,
) -> dict[str, object]:
    """Fit primary/sensitivity t-SNE embeddings and save outputs."""

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    cache_dir = output_dir / "sweep_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_input, pre_pca_info = prepare_tsne_input(
        X,
        pre_pca_components=pre_pca_components,
        random_state=random_state,
    )
    X_original = X.to_numpy(dtype=float, copy=False)

    # Include the primary perplexity and metric exactly once, even if the user
    # omitted either one from the corresponding sensitivity grid.
    all_perplexities = sorted(
        set(float(v) for v in list(perplexity_grid) + [float(perplexity)])
    )
    if not all_perplexities:
        all_perplexities = [float(perplexity)]

    all_metrics = list(dict.fromkeys([str(v) for v in metric_grid] + [str(metric)]))
    if not all_metrics:
        all_metrics = [str(metric)]

    embeddings: dict[tuple[str, float], np.ndarray] = {}
    models: dict[tuple[str, float], TSNE] = {}
    diagnostic_rows: list[dict[str, object]] = []
    cached_score_tables: list[pd.DataFrame] = []

    for metric_name in all_metrics:
        for p in all_perplexities:
            model, embedding = fit_tsne(
                X_input,
                perplexity=p,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                max_iter=max_iter,
                init=init,
                metric=metric_name,
                method=method,
                angle=angle,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
            )
            diagnostics = embedding_diagnostics(
                X_original,
                embedding,
                model,
                original_metric=metric_name,
                neighborhood_k=neighborhood_k,
                n_jobs=n_jobs,
            )
            diagnostic_rows.append(
                {"metric": metric_name, "perplexity": p, **diagnostics}
            )
            embeddings[(metric_name, p)] = embedding
            models[(metric_name, p)] = model

            cached_scores = pd.DataFrame(
                {
                    "sample_id": X.index,
                    "metric": metric_name,
                    "perplexity": p,
                    "tSNE1": embedding[:, 0],
                    "tSNE2": embedding[:, 1],
                }
            )
            cached_score_tables.append(cached_scores)
            p_tag = f"{p:g}".replace(".", "p")
            cached_scores.to_csv(
                cache_dir
                / f"tsne_{_safe_name(metric_name)}_perplexity_{p_tag}.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )

    diagnostic_table = pd.DataFrame(diagnostic_rows).sort_values(
        ["metric", "perplexity"]
    )
    diagnostic_table.to_csv(
        output_dir / "tsne_sensitivity_diagnostics.tsv",
        sep="\t",
        index=False,
    )

    # Retain the original perplexity-only diagnostics output for downstream
    # compatibility, filtered to the primary metric.
    primary_metric_diagnostics = (
        diagnostic_table.loc[diagnostic_table["metric"].eq(metric)]
        .sort_values("perplexity")
        .reset_index(drop=True)
    )
    primary_metric_diagnostics.drop(columns="metric").to_csv(
        output_dir / "tsne_perplexity_diagnostics.tsv",
        sep="\t",
        index=False,
    )

    all_cached_scores = pd.concat(cached_score_tables, ignore_index=True)
    all_cached_scores.to_csv(
        output_dir / "tsne_sensitivity_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    primary_key = (str(metric), float(perplexity))
    primary_embedding = embeddings[primary_key]
    primary_model = models[primary_key]

    scores = pd.DataFrame(
        primary_embedding,
        index=X.index,
        columns=["tSNE1", "tSNE2"],
    )
    scores.index.name = "sample_id"
    scores.to_csv(output_dir / "tsne_scores.tsv.gz", sep="\t", compression="gzip")

    scores_with_metadata = (
        scores.reset_index()
        .merge(sample_info, on="sample_id", how="left", validate="one_to_one", sort=False)
    )
    scores_with_metadata.to_csv(
        output_dir / "tsne_scores_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    plot_embedding(
        scores_with_metadata,
        figure_dir / "tsne1_tsne2.png",
        perplexity=perplexity,
        metric=metric,
    )

    requested_metadata = list(dict.fromkeys(color_by))
    for metadata_column in requested_metadata:
        if metadata_column not in scores_with_metadata.columns:
            available = ", ".join(sample_info.columns)
            raise KeyError(
                f"Cannot color t-SNE by {metadata_column!r}; column not found. "
                f"Available metadata columns: {available}"
            )
        plot_embedding(
            scores_with_metadata,
            figure_dir / f"tsne1_tsne2_by_{_safe_name(metadata_column)}.png",
            color_by=metadata_column,
            perplexity=perplexity,
            metric=metric,
        )

    primary_metric_embeddings = {
        p: embeddings[(str(metric), p)] for p in all_perplexities
    }
    if len(primary_metric_embeddings) > 1:
        plot_perplexity_sweep(
            primary_metric_embeddings,
            figure_dir / "tsne_perplexity_sweep.png",
        )
        plot_perplexity_diagnostics(
            primary_metric_diagnostics,
            figure_dir / "tsne_perplexity_diagnostics.png",
            neighborhood_k=neighborhood_k,
        )

    primary_perplexity_embeddings = {
        metric_name: embeddings[(metric_name, float(perplexity))]
        for metric_name in all_metrics
    }
    if len(primary_perplexity_embeddings) > 1:
        plot_metric_sweep(
            primary_perplexity_embeddings,
            figure_dir / "tsne_metric_sweep.png",
            perplexity=perplexity,
        )

    if len(embeddings) > 1:
        plot_sensitivity_sweep(
            embeddings,
            figure_dir / "tsne_metric_perplexity_sweep.png",
            metrics=all_metrics,
            perplexities=all_perplexities,
        )

    primary_row = diagnostic_table.loc[
        diagnostic_table["metric"].eq(metric)
        & np.isclose(diagnostic_table["perplexity"], float(perplexity))
    ].iloc[0]

    summary: dict[str, object] = {
        "input": {
            "samples": int(X.shape[0]),
            "genes": int(X.shape[1]),
            **pre_pca_info,
        },
        "tsne": {
            "n_components": 2,
            "perplexity": float(perplexity),
            "early_exaggeration": float(early_exaggeration),
            "learning_rate_requested": learning_rate,
            "effective_learning_rate": (
                float(getattr(primary_model, "learning_rate_"))
                if hasattr(primary_model, "learning_rate_")
                else None
            ),
            "max_iter": int(max_iter),
            "n_iter": int(getattr(primary_model, "n_iter_", -1)),
            "init": init,
            "metric": metric,
            "method": method,
            "angle": float(angle),
            "random_state": int(random_state),
            "kl_divergence": float(primary_model.kl_divergence_),
            **{
                col: float(primary_row[col])
                for col in diagnostic_table.columns
                if col.startswith("trustworthiness_") or col.startswith("mean_jaccard_")
            },
        },
        "sensitivity": {
            "perplexities": [float(v) for v in all_perplexities],
            "metrics": all_metrics,
            "neighborhood_k": [int(v) for v in neighborhood_k],
            "n_embeddings": int(len(embeddings)),
        },
        "outputs": {
            "metadata_overlays": requested_metadata,
            "sensitivity_scores": "tsne_sensitivity_scores.tsv.gz",
            "sensitivity_diagnostics": "tsne_sensitivity_diagnostics.tsv",
            "sweep_embedding_dir": "sweep_embeddings",
        },
    }

    with (output_dir / "tsne_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def format_summary(summary: dict[str, object]) -> str:
    """Format the primary t-SNE result for terminal output."""

    input_info = summary["input"]
    tsne_info = summary["tsne"]
    sensitivity = summary["sensitivity"]

    lines = [
        "TCGA STAD t-SNE summary",
        "-----------------------",
        f"Samples:                         {input_info['samples']:,}",
        f"Genes:                           {input_info['genes']:,}",
        f"Pre-PCA:                         {'yes' if input_info['pre_pca_enabled'] else 'no'}",
    ]
    if input_info["pre_pca_enabled"]:
        lines.extend(
            [
                f"Pre-PCA components:              {input_info['pre_pca_components']}",
                f"Pre-PCA variance retained:       {input_info['pre_pca_variance_percent']:.2f}%",
            ]
        )

    lines.extend(
        [
            f"Perplexity:                      {tsne_info['perplexity']:g}",
            f"Metric:                          {tsne_info['metric']}",
            f"Initialization:                  {tsne_info['init']}",
            f"Early exaggeration:              {tsne_info['early_exaggeration']:g}",
            f"Learning rate (effective):       {tsne_info['effective_learning_rate']}",
            f"Iterations:                      {tsne_info['n_iter']}",
            f"KL divergence:                   {tsne_info['kl_divergence']:.4f}",
        ]
    )

    for k in sensitivity["neighborhood_k"]:
        trust = tsne_info.get(f"trustworthiness_k{k}")
        jaccard = tsne_info.get(f"mean_jaccard_k{k}")
        if trust is not None:
            lines.append(f"Trustworthiness (k={k:<2}):          {trust:.4f}")
        if jaccard is not None:
            lines.append(f"Mean kNN Jaccard (k={k:<2}):         {jaccard:.4f}")

    lines.append(
        "Perplexity sweep:               "
        + ", ".join(f"{v:g}" for v in sensitivity["perplexities"])
    )
    lines.append(
        "Metric sweep:                   "
        + ", ".join(str(v) for v in sensitivity["metrics"])
    )
    lines.append(
        f"Sensitivity embeddings cached:  {sensitivity['n_embeddings']}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run t-SNE on the canonical TCGA STAD dimensionality-reduction matrix."
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Directory created by preprocessing.py (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/tsne",
        help="Directory for t-SNE outputs (default: analysis/tsne)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Primary t-SNE perplexity (default: 30)",
    )
    parser.add_argument(
        "--perplexity-grid",
        type=float,
        nargs="*",
        default=list(DEFAULT_PERPLEXITY_GRID),
        help=(
            "Perplexities used for sensitivity diagnostics (default: 5 15 30 50). "
            "Pass --perplexity-grid with no values to run only the primary perplexity."
        ),
    )
    parser.add_argument(
        "--early-exaggeration",
        type=float,
        default=12.0,
        help="Early-exaggeration factor (default: 12)",
    )
    parser.add_argument(
        "--learning-rate",
        type=parse_learning_rate,
        default="auto",
        help="Learning rate: 'auto' or a positive number (default: auto)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--init",
        choices=("pca", "random"),
        default="pca",
        help="Embedding initialization (default: pca)",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="Input-space distance metric (default: euclidean)",
    )
    parser.add_argument(
        "--metric-grid",
        nargs="*",
        default=list(DEFAULT_METRIC_GRID),
        help=(
            "Distance metrics used for sensitivity diagnostics "
            "(default: euclidean cosine correlation). Pass --metric-grid with "
            "no values to run only the primary --metric."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("barnes_hut", "exact"),
        default="barnes_hut",
        help="t-SNE optimization method (default: barnes_hut)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.5,
        help="Barnes-Hut accuracy/speed tradeoff (default: 0.5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for neighbor search (default: -1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--pre-pca-components",
        type=int,
        default=0,
        help=(
            "Optional PCA dimensions before t-SNE; 0 disables pre-PCA so the canonical "
            "X_dimred matrix is used directly (default: 0)"
        ),
    )
    parser.add_argument(
        "--neighborhood-k",
        type=int,
        nargs="*",
        default=list(DEFAULT_NEIGHBORHOOD_K),
        help="k values for local-neighborhood diagnostics (default: 15 30)",
    )
    parser.add_argument(
        "--color-by",
        nargs="*",
        default=list(DEFAULT_COLOR_BY),
        help=(
            "Metadata columns used for t-SNE overlay plots. Default: center_code. "
            "Example: --color-by center_code gender race histological_grade"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="scikit-learn t-SNE verbosity (default: 0)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.pre_pca_components < 0:
        raise ValueError("--pre-pca-components cannot be negative")
    if not args.neighborhood_k:
        raise ValueError("--neighborhood-k must contain at least one value")
    if any(k < 1 for k in args.neighborhood_k):
        raise ValueError("all --neighborhood-k values must be >= 1")

    X, sample_info = load_inputs(args.processed_dir)
    summary = run_analysis(
        X,
        sample_info,
        output_dir=args.output_dir,
        perplexity=args.perplexity,
        perplexity_grid=args.perplexity_grid,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        init=args.init,
        metric=args.metric,
        metric_grid=args.metric_grid,
        method=args.method,
        angle=args.angle,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        pre_pca_components=args.pre_pca_components,
        neighborhood_k=args.neighborhood_k,
        color_by=args.color_by,
        verbose=args.verbose,
    )

    print(format_summary(summary))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()