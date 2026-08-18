#!/usr/bin/env python3
"""UMAP analysis for the rebuilt TCGA STAD project.

This script consumes the canonical ``X_dimred`` matrix produced by
``preprocessing.py`` and computes a two-dimensional Uniform Manifold
Approximation and Projection (UMAP) embedding. No additional gene filtering,
scaling, clustering, or metadata-driven parameter selection is performed.

Primary analysis
----------------
The canonical embedding uses parameters that match the baseline UMAP setup in
the project reference manuscript and the standard umap-learn defaults:

* n_components = 2
* n_neighbors = 15
* min_dist = 0.1
* metric = ``euclidean``
* init = ``spectral``
* random_state = 42

The script intentionally keeps clustering and formal metadata-association tests
out of this stage so PCA, MDS, t-SNE, and UMAP can later be compared with one
common downstream evaluation workflow.

Sensitivity analyses
--------------------
UMAP can change substantially with its neighborhood size, minimum embedding
spacing, and input-space distance metric. The script therefore supports three
small one-factor-at-a-time sweeps while holding the other primary parameters
fixed:

* ``n_neighbors``: 5, 15, 30, 50
* ``min_dist``: 0.0, 0.1, 0.5, 1.0
* ``metric``: euclidean, cosine, canberra

The metric sweep mirrors the three representative distance metrics explored in
the project reference manuscript. The original class UMAP notebook used cosine
as its primary metric, so keeping cosine in the sensitivity analysis also lets
us compare the rebuilt workflow with the historical analysis.

Embedding diagnostics
---------------------
``trustworthiness``
    Measures whether neighbors in the low-dimensional embedding were also
    neighbors in the high-dimensional input. Values approach 1 as local
    neighborhood preservation improves.

``mean Jaccard kNN overlap``
    For each sample, compares its k nearest-neighbor set in the original space
    with its k nearest-neighbor set in the UMAP embedding, then averages the
    Jaccard index across samples. Default k values are 15 and 30.

For metric sensitivity runs, these neighborhood diagnostics use the same input
metric as the UMAP fit. Values from different metrics therefore answer slightly
different neighborhood questions and should not be treated as a single
optimization leaderboard.

Optional PCA pre-reduction
--------------------------
``--pre-pca-components`` can reduce the 8,194-gene matrix to a specified number
of principal components before UMAP. It is disabled by default so PCA, MDS,
t-SNE, and UMAP all consume the same canonical matrix in the primary comparison.
It is best treated as a later sensitivity/performance analysis.

Expected inputs in ``--processed-dir``
--------------------------------------
X_dimred.tsv.gz
    Samples x genes canonical dimensionality-reduction matrix.
sample_info.tsv
    Sample identifiers plus patient, technical, and clinical metadata.

Requires
--------
``umap-learn`` (imported as ``umap``).

Run directly with:

python umap_analysis.py \
    --processed-dir processed \
    --output-dir analysis/umap \
    --color-by \
        center_code \
        gender \
        race \
        ajcc_pathologic_tumor_stage \
        histological_grade \
        vital_status

Sensitivity Analysis: Trustworthiness @ k=15, Trustworthiness @ k=30, Mean kNN Jaccard @ k=15, Mean kNN Jaccard @ k=30
n_neighbors:
5, 15, 30, 50

min_dist:
0.0, 0.1, 0.5, 1.0

metric:
euclidean, cosine, canberra

Optional (off by default, like with tsne.py): --pre-pca-components 50
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
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "UMAP analysis requires the 'umap-learn' package. Install it with "
        "`pip install umap-learn` or `conda install -c conda-forge umap-learn`."
    ) from exc


DEFAULT_COLOR_BY = ("center_code",)
DEFAULT_N_NEIGHBORS_GRID = (5, 15, 30, 50)
DEFAULT_MIN_DIST_GRID = (0.0, 0.1, 0.5, 1.0)
DEFAULT_METRIC_GRID = ("euclidean", "cosine", "canberra")
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

    sample_info = (
        sample_info.set_index("sample_id")
        .loc[X.index]
        .reset_index()
    )

    values = X.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("X_dimred contains NaN or infinite values")

    return X, sample_info


def optional_pre_pca(
    X: pd.DataFrame,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optionally reduce the canonical matrix with PCA before UMAP."""

    values = X.to_numpy(dtype=float, copy=False)
    if n_components <= 0:
        return values, {
            "pre_pca_enabled": False,
            "pre_pca_components": 0,
            "pre_pca_variance_percent": None,
        }

    max_components = min(X.shape[0] - 1, X.shape[1])
    if n_components > max_components:
        raise ValueError(
            f"--pre-pca-components cannot exceed {max_components} for this matrix"
        )

    model = PCA(
        n_components=n_components,
        svd_solver="auto",
        random_state=random_state,
    )
    reduced = model.fit_transform(values)
    variance = float(model.explained_variance_ratio_.sum() * 100.0)
    return reduced, {
        "pre_pca_enabled": True,
        "pre_pca_components": int(n_components),
        "pre_pca_variance_percent": variance,
    }


def fit_umap(
    X_input: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    init: str,
    n_epochs: int | None,
    random_state: int,
    n_jobs: int,
    verbose: bool = False,
) -> tuple[object, np.ndarray]:
    """Fit one two-dimensional UMAP embedding."""

    n_samples = X_input.shape[0]
    if not 2 <= n_neighbors < n_samples:
        raise ValueError(
            f"n_neighbors must be between 2 and n_samples-1 ({n_samples - 1}); "
            f"got {n_neighbors}"
        )
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("min_dist must be between 0 and 1")

    kwargs: dict[str, object] = {
        "n_components": 2,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "init": init,
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbose": bool(verbose),
    }
    if n_epochs is not None:
        kwargs["n_epochs"] = int(n_epochs)

    model = umap.UMAP(**kwargs)
    embedding = model.fit_transform(X_input)
    if not np.isfinite(embedding).all():
        raise ValueError("UMAP produced NaN or infinite coordinates")
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
            fallback = NearestNeighbors(
                n_neighbors=min(X.shape[0], n_neighbors + 2),
                metric=metric,
                n_jobs=n_jobs,
            ).fit(X)
            expanded = fallback.kneighbors(
                X[i : i + 1], return_distance=False
            )[0]
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
    """Mean Jaccard overlap of original-space and UMAP-space kNN sets."""

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


def metric_aware_trustworthiness(
    X_original: np.ndarray,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    original_metric: str,
) -> float:
    """Compute trustworthiness using the UMAP input metric when supported."""

    if n_neighbors >= X_original.shape[0] / 2:
        raise ValueError(
            f"trustworthiness requires k < n_samples/2; got k={n_neighbors}"
        )

    params = inspect.signature(trustworthiness).parameters
    kwargs: dict[str, object] = {"n_neighbors": int(n_neighbors)}
    if "metric" in params:
        kwargs["metric"] = original_metric
    return float(trustworthiness(X_original, embedding, **kwargs))


def embedding_diagnostics(
    X_original: np.ndarray,
    embedding: np.ndarray,
    *,
    original_metric: str,
    neighborhood_k: Sequence[int],
    n_jobs: int,
) -> dict[str, float]:
    """Compute local-neighborhood preservation diagnostics."""

    diagnostics: dict[str, float] = {}
    for k in neighborhood_k:
        diagnostics[f"trustworthiness_k{k}"] = metric_aware_trustworthiness(
            X_original,
            embedding,
            n_neighbors=k,
            original_metric=original_metric,
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
    """Treat only true numeric dtypes as continuous metadata."""

    return pd.api.types.is_numeric_dtype(series)


def _safe_name(value: object) -> str:
    """Convert a value to a filesystem-safe filename fragment."""

    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def plot_embedding(
    data: pd.DataFrame,
    output_path: Path,
    *,
    color_by: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Save a two-dimensional UMAP scatterplot."""

    fig, ax = plt.subplots(figsize=(7, 6))

    if color_by is None:
        ax.scatter(data["UMAP1"], data["UMAP2"], s=28, alpha=0.8)
    else:
        if color_by not in data.columns:
            raise KeyError(f"Metadata column not found: {color_by}")

        series = data[color_by]
        valid = series.notna()

        if _is_numeric_metadata(series):
            numeric = pd.to_numeric(series, errors="coerce")
            scatter = ax.scatter(
                data.loc[valid, "UMAP1"],
                data.loc[valid, "UMAP2"],
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
                    data.loc[mask, "UMAP1"],
                    data.loc[mask, "UMAP2"],
                    s=28,
                    alpha=0.8,
                    label=category,
                )

        missing = ~valid
        if missing.any():
            ax.scatter(
                data.loc[missing, "UMAP1"],
                data.loc[missing, "UMAP2"],
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

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    title = "UMAP embedding"
    if title_suffix:
        title += f" — {title_suffix}"
    if color_by is not None:
        title += f" — {color_by}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(
    fits: list[tuple[str, np.ndarray]],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a grid of embeddings for one sensitivity sweep."""

    if not fits:
        return
    n = len(fits)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for ax, (label, embedding) in zip(axes.flat, fits):
        ax.scatter(embedding[:, 0], embedding[:, 1], s=18, alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    for ax in axes.flat[len(fits) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_run_configs(
    *,
    primary_n_neighbors: int,
    primary_min_dist: float,
    primary_metric: str,
    n_neighbors_grid: Sequence[int],
    min_dist_grid: Sequence[float],
    metric_grid: Sequence[str],
) -> list[dict[str, object]]:
    """Build deduplicated primary and one-factor-at-a-time sensitivity configs."""

    configs: list[dict[str, object]] = [
        {
            "sweep": "primary",
            "label": "primary",
            "n_neighbors": int(primary_n_neighbors),
            "min_dist": float(primary_min_dist),
            "metric": str(primary_metric),
        }
    ]

    for value in n_neighbors_grid:
        configs.append(
            {
                "sweep": "n_neighbors",
                "label": f"n_neighbors={int(value)}",
                "n_neighbors": int(value),
                "min_dist": float(primary_min_dist),
                "metric": str(primary_metric),
            }
        )

    for value in min_dist_grid:
        configs.append(
            {
                "sweep": "min_dist",
                "label": f"min_dist={float(value):g}",
                "n_neighbors": int(primary_n_neighbors),
                "min_dist": float(value),
                "metric": str(primary_metric),
            }
        )

    for value in metric_grid:
        configs.append(
            {
                "sweep": "metric",
                "label": f"metric={value}",
                "n_neighbors": int(primary_n_neighbors),
                "min_dist": float(primary_min_dist),
                "metric": str(value),
            }
        )

    # Deduplicate computationally identical fits while preserving all sweep labels.
    unique: dict[tuple[int, float, str], dict[str, object]] = {}
    for cfg in configs:
        key = (
            int(cfg["n_neighbors"]),
            float(cfg["min_dist"]),
            str(cfg["metric"]),
        )
        if key not in unique:
            unique[key] = {
                "n_neighbors": key[0],
                "min_dist": key[1],
                "metric": key[2],
                "memberships": [],
            }
        unique[key]["memberships"].append(
            {"sweep": cfg["sweep"], "label": cfg["label"]}
        )

    return list(unique.values())


def run_umap_analysis(
    *,
    X: pd.DataFrame,
    sample_info: pd.DataFrame,
    output_dir: str | Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    init: str,
    n_epochs: int | None,
    random_state: int,
    n_jobs: int,
    pre_pca_components: int,
    neighborhood_k: Sequence[int],
    n_neighbors_grid: Sequence[int],
    min_dist_grid: Sequence[float],
    metric_grid: Sequence[str],
    color_by: Iterable[str],
    verbose: bool,
) -> dict[str, object]:
    """Fit primary/sensitivity UMAPs, save outputs, and return summary."""

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    X_input, pre_pca_info = optional_pre_pca(
        X,
        n_components=pre_pca_components,
        random_state=random_state,
    )

    configs = build_run_configs(
        primary_n_neighbors=n_neighbors,
        primary_min_dist=min_dist,
        primary_metric=metric,
        n_neighbors_grid=n_neighbors_grid,
        min_dist_grid=min_dist_grid,
        metric_grid=metric_grid,
    )

    rows: list[dict[str, object]] = []
    fits: dict[tuple[int, float, str], tuple[object, np.ndarray]] = {}

    for cfg in configs:
        nn = int(cfg["n_neighbors"])
        md = float(cfg["min_dist"])
        mt = str(cfg["metric"])
        print(f"Fitting UMAP: n_neighbors={nn}, min_dist={md:g}, metric={mt}")

        model, embedding = fit_umap(
            X_input,
            n_neighbors=nn,
            min_dist=md,
            metric=mt,
            init=init,
            n_epochs=n_epochs,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        key = (nn, md, mt)
        fits[key] = (model, embedding)
        diagnostics = embedding_diagnostics(
            X_input,
            embedding,
            original_metric=mt,
            neighborhood_k=neighborhood_k,
            n_jobs=n_jobs,
        )

        for membership in cfg["memberships"]:
            rows.append(
                {
                    "sweep": membership["sweep"],
                    "label": membership["label"],
                    "n_neighbors": nn,
                    "min_dist": md,
                    "metric": mt,
                    **diagnostics,
                }
            )

    diagnostic_table = pd.DataFrame(rows)
    diagnostic_table.to_csv(
        output_dir / "umap_parameter_diagnostics.tsv",
        sep="\t",
        index=False,
    )

    primary_key = (int(n_neighbors), float(min_dist), str(metric))
    primary_model, primary_embedding = fits[primary_key]

    scores = pd.DataFrame(
        primary_embedding,
        index=X.index,
        columns=["UMAP1", "UMAP2"],
    )
    scores.index.name = "sample_id"
    scores.to_csv(
        output_dir / "umap_scores.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    scores_with_metadata = (
        scores.reset_index()
        .merge(sample_info, on="sample_id", how="left", validate="one_to_one", sort=False)
    )
    scores_with_metadata.to_csv(
        output_dir / "umap_scores_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    plot_embedding(
        scores_with_metadata,
        figure_dir / "umap1_umap2.png",
        title_suffix=(
            f"n_neighbors={n_neighbors}, min_dist={min_dist:g}, metric={metric}"
        ),
    )

    requested_metadata = list(dict.fromkeys(color_by))
    for metadata_column in requested_metadata:
        if metadata_column not in scores_with_metadata.columns:
            available = ", ".join(sample_info.columns)
            raise KeyError(
                f"Cannot color UMAP by {metadata_column!r}; column not found. "
                f"Available metadata columns: {available}"
            )
        plot_embedding(
            scores_with_metadata,
            figure_dir / f"umap1_umap2_by_{_safe_name(metadata_column)}.png",
            color_by=metadata_column,
            title_suffix=(
                f"n_neighbors={n_neighbors}, min_dist={min_dist:g}, metric={metric}"
            ),
        )

    # Sweep figures use the already-cached embeddings.
    neighbor_fits: list[tuple[str, np.ndarray]] = []
    for value in n_neighbors_grid:
        key = (int(value), float(min_dist), str(metric))
        if key in fits:
            neighbor_fits.append((f"n_neighbors={int(value)}", fits[key][1]))
    plot_sweep(
        neighbor_fits,
        figure_dir / "umap_n_neighbors_sweep.png",
        title="UMAP n_neighbors sensitivity",
    )

    min_dist_fits: list[tuple[str, np.ndarray]] = []
    for value in min_dist_grid:
        key = (int(n_neighbors), float(value), str(metric))
        if key in fits:
            min_dist_fits.append((f"min_dist={float(value):g}", fits[key][1]))
    plot_sweep(
        min_dist_fits,
        figure_dir / "umap_min_dist_sweep.png",
        title="UMAP min_dist sensitivity",
    )

    metric_fits: list[tuple[str, np.ndarray]] = []
    for value in metric_grid:
        key = (int(n_neighbors), float(min_dist), str(value))
        if key in fits:
            metric_fits.append((f"metric={value}", fits[key][1]))
    plot_sweep(
        metric_fits,
        figure_dir / "umap_metric_sweep.png",
        title="UMAP metric sensitivity",
    )

    primary_row = diagnostic_table.loc[
        (diagnostic_table["sweep"] == "primary")
        & (diagnostic_table["n_neighbors"] == int(n_neighbors))
        & np.isclose(diagnostic_table["min_dist"], float(min_dist))
        & (diagnostic_table["metric"] == str(metric))
    ].iloc[0]

    summary: dict[str, object] = {
        "input": {
            "samples": int(X.shape[0]),
            "genes": int(X.shape[1]),
            **pre_pca_info,
        },
        "umap": {
            "package_version": getattr(umap, "__version__", None),
            "n_components": 2,
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
            "init": init,
            "n_epochs_requested": n_epochs,
            "random_state": int(random_state),
            "n_jobs_requested": int(n_jobs),
            **{
                col: float(primary_row[col])
                for col in diagnostic_table.columns
                if col.startswith("trustworthiness_")
                or col.startswith("mean_jaccard_")
            },
        },
        "sensitivity": {
            "n_neighbors": [int(v) for v in n_neighbors_grid],
            "min_dist": [float(v) for v in min_dist_grid],
            "metrics": [str(v) for v in metric_grid],
            "neighborhood_k": [int(v) for v in neighborhood_k],
        },
        "outputs": {
            "metadata_overlays": requested_metadata,
        },
    }

    with (output_dir / "umap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def format_summary(summary: dict[str, object]) -> str:
    """Format the primary UMAP result for terminal output."""

    input_info = summary["input"]
    umap_info = summary["umap"]
    sensitivity = summary["sensitivity"]

    lines = [
        "TCGA STAD UMAP summary",
        "----------------------",
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
            f"n_neighbors:                     {umap_info['n_neighbors']}",
            f"min_dist:                        {umap_info['min_dist']:g}",
            f"Metric:                          {umap_info['metric']}",
            f"Initialization:                  {umap_info['init']}",
            f"Random state:                    {umap_info['random_state']}",
        ]
    )

    for k in sensitivity["neighborhood_k"]:
        trust = umap_info.get(f"trustworthiness_k{k}")
        jaccard = umap_info.get(f"mean_jaccard_k{k}")
        if trust is not None:
            lines.append(f"Trustworthiness (k={k:<2}):          {trust:.4f}")
        if jaccard is not None:
            lines.append(f"Mean kNN Jaccard (k={k:<2}):         {jaccard:.4f}")

    lines.extend(
        [
            "n_neighbors sweep:               "
            + ", ".join(str(v) for v in sensitivity["n_neighbors"]),
            "min_dist sweep:                  "
            + ", ".join(f"{v:g}" for v in sensitivity["min_dist"]),
            "metric sweep:                    "
            + ", ".join(sensitivity["metrics"]),
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UMAP on the canonical TCGA STAD dimensionality-reduction matrix."
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Directory created by preprocessing.py (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/umap",
        help="Directory for UMAP outputs (default: analysis/umap)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Primary UMAP neighborhood size (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Primary UMAP minimum embedding distance (default: 0.1)",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="Primary input-space distance metric (default: euclidean)",
    )
    parser.add_argument(
        "--init",
        default="spectral",
        choices=("spectral", "random"),
        help="UMAP initialization (default: spectral)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Optional number of UMAP optimization epochs; default lets umap-learn choose",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible UMAP initialization/optimization (default: 42)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel jobs for UMAP neighbor search (default: 1). A fixed random_state "
            "is used for reproducibility; current umap-learn may restrict parallelism "
            "when a seed is set."
        ),
    )
    parser.add_argument(
        "--pre-pca-components",
        type=int,
        default=0,
        help=(
            "Optional PCA dimensions before UMAP; 0 disables pre-PCA so the canonical "
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
        "--n-neighbors-grid",
        type=int,
        nargs="*",
        default=list(DEFAULT_N_NEIGHBORS_GRID),
        help=(
            "n_neighbors values for sensitivity analysis (default: 5 15 30 50). "
            "Pass --n-neighbors-grid with no values to disable this sweep."
        ),
    )
    parser.add_argument(
        "--min-dist-grid",
        type=float,
        nargs="*",
        default=list(DEFAULT_MIN_DIST_GRID),
        help=(
            "min_dist values for sensitivity analysis (default: 0 0.1 0.5 1). "
            "Pass --min-dist-grid with no values to disable this sweep."
        ),
    )
    parser.add_argument(
        "--metric-grid",
        nargs="*",
        default=list(DEFAULT_METRIC_GRID),
        help=(
            "Metrics for sensitivity analysis (default: euclidean cosine canberra). "
            "Pass --metric-grid with no values to disable this sweep."
        ),
    )
    parser.add_argument(
        "--color-by",
        nargs="*",
        default=list(DEFAULT_COLOR_BY),
        help=(
            "Metadata columns used for UMAP overlay plots. Default: center_code. "
            "Example: --color-by center_code gender race histological_grade"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable umap-learn progress output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_epochs is not None and args.n_epochs < 1:
        raise ValueError("--n-epochs must be positive")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be 0")
    if not args.neighborhood_k:
        raise ValueError("At least one --neighborhood-k value is required")
    if any(k < 1 for k in args.neighborhood_k):
        raise ValueError("All --neighborhood-k values must be >= 1")

    X, sample_info = load_inputs(args.processed_dir)
    summary = run_umap_analysis(
        X=X,
        sample_info=sample_info,
        output_dir=args.output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        init=args.init,
        n_epochs=args.n_epochs,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        pre_pca_components=args.pre_pca_components,
        neighborhood_k=args.neighborhood_k,
        n_neighbors_grid=args.n_neighbors_grid,
        min_dist_grid=args.min_dist_grid,
        metric_grid=args.metric_grid,
        color_by=args.color_by,
        verbose=args.verbose,
    )

    print(format_summary(summary))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()