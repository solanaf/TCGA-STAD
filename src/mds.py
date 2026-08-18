#!/usr/bin/env python3
"""Multidimensional scaling for the rebuilt TCGA STAD project.

This script consumes the canonical ``X_dimred`` matrix produced by
``preprocessing.py`` and performs multidimensional scaling (MDS) without any
additional gene filtering or scaling.

The primary analysis is two-dimensional *metric* MDS using Euclidean
sample-to-sample distances. This matches the baseline MDS formulation used in
the project manuscript while ensuring that PCA, MDS, t-SNE, and UMAP all see
the same preprocessed expression matrix.

MDS is evaluated by how well the low-dimensional embedding preserves the
original pairwise sample dissimilarities. The script therefore reports:

* normalized Kruskal Stress-1 (lower is better),
* Pearson correlation of original vs embedded pairwise distances,
* Spearman correlation of original vs embedded pairwise distances,
* a Shepard plot of original vs embedded distances,
* an optional stress-by-dimension analysis (default: 2, 3, and 5 dimensions).

Clustering and formal metadata-association testing are intentionally omitted.
Those will be applied later as a common downstream evaluation step to PCA,
MDS, t-SNE, and UMAP embeddings.

Expected inputs in ``--processed-dir``
--------------------------------------
X_dimred.tsv.gz
    Samples x genes canonical dimensionality-reduction matrix.
sample_info.tsv
    Sample identifiers plus patient, technical, and clinical metadata.

Run directly with:

python mds.py \
    --processed-dir processed \
    --output-dir analysis/mds \
    --color-by \
        center_code \
        gender \
        race \
        ajcc_pathologic_tumor_stage \
        histological_grade \
        vital_status


Previous version:

python mds.py \
    --processed-dir processed \
    --output-dir analysis/mds_nonmetric \
    --distance-metric correlation \
    --nonmetric \
    --primary-dimensions 5
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS


DEFAULT_COLOR_BY = ("center_code",)
DEFAULT_DIMENSION_GRID = (2, 3, 5)


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


def compute_dissimilarities(X: pd.DataFrame, metric: str) -> np.ndarray:
    """Compute a symmetric sample-to-sample dissimilarity matrix.

    ``scipy.spatial.distance.pdist`` is used so the same precomputed matrix can
    be reused for every dimensionality in the stress curve.
    """

    if metric == "precomputed":
        raise ValueError("'precomputed' is not a valid --distance-metric here")

    try:
        condensed = pdist(X.to_numpy(dtype=float, copy=False), metric=metric)
    except Exception as exc:
        raise ValueError(
            f"Could not compute pairwise distances with metric {metric!r}: {exc}"
        ) from exc

    if not np.isfinite(condensed).all():
        raise ValueError(
            f"Distance metric {metric!r} produced NaN or infinite values."
        )

    D = squareform(condensed)
    np.fill_diagonal(D, 0.0)
    return D


def _build_mds_model(
    *,
    n_components: int,
    metric_mds: bool,
    n_init: int,
    max_iter: int,
    eps: float,
    n_jobs: int,
    random_state: int,
) -> tuple[MDS, bool]:
    """Construct an MDS estimator across old and new scikit-learn APIs.

    scikit-learn 1.8 renamed the old boolean ``metric`` argument to
    ``metric_mds`` and repurposed ``metric`` for the distance metric. We always
    pass a precomputed distance matrix and adapt to either API.

    Returns
    -------
    model
        Configured MDS estimator.
    normalized_stress_requested
        Whether the installed scikit-learn supports requesting Stress-1
        directly.
    """

    signature = inspect.signature(MDS)
    params = signature.parameters

    kwargs: dict[str, object] = {
        "n_components": n_components,
        "n_init": n_init,
        "max_iter": max_iter,
        "eps": eps,
        "n_jobs": n_jobs,
        "random_state": random_state,
    }

    if "metric_mds" in params:
        kwargs["metric_mds"] = metric_mds
        kwargs["metric"] = "precomputed"
    else:
        # scikit-learn <= 1.7 API
        kwargs["metric"] = metric_mds
        kwargs["dissimilarity"] = "precomputed"

    normalized_supported = "normalized_stress" in params
    if normalized_supported:
        kwargs["normalized_stress"] = True

    # Keep random initialization when the newer API exposes this option. This
    # is closest to the historical sklearn MDS behavior and makes n_init useful.
    if "init" in params:
        kwargs["init"] = "random"

    return MDS(**kwargs), normalized_supported


def _manual_metric_stress1(
    raw_stress: float,
    embedding: np.ndarray,
) -> float:
    """Convert raw metric MDS stress to scikit-learn's Stress-1 definition."""

    embedded_distances = pdist(embedding, metric="euclidean")
    denominator = float(np.square(embedded_distances).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(raw_stress / denominator))


def fit_mds(
    D: np.ndarray,
    *,
    n_components: int,
    metric_mds: bool,
    n_init: int,
    max_iter: int,
    eps: float,
    n_jobs: int,
    random_state: int,
) -> tuple[MDS, np.ndarray, float]:
    """Fit MDS to a precomputed dissimilarity matrix and return Stress-1."""

    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    if n_components >= D.shape[0]:
        raise ValueError("n_components must be smaller than the number of samples")

    model, normalized_requested = _build_mds_model(
        n_components=n_components,
        metric_mds=metric_mds,
        n_init=n_init,
        max_iter=max_iter,
        eps=eps,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    embedding = model.fit_transform(D)

    if normalized_requested:
        stress1 = float(model.stress_)
    elif metric_mds:
        stress1 = _manual_metric_stress1(float(model.stress_), embedding)
    else:
        # Older sklearn versions do not expose fitted non-metric disparities,
        # so a faithful normalized non-metric stress cannot be reconstructed.
        stress1 = float("nan")

    return model, embedding, stress1


def distance_preservation(
    D: np.ndarray,
    embedding: np.ndarray,
) -> dict[str, float]:
    """Measure preservation of pairwise distances in the MDS embedding."""

    original = squareform(D, checks=False)
    embedded = pdist(embedding, metric="euclidean")

    pearson = pearsonr(original, embedded)
    spearman = spearmanr(original, embedded)

    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def build_dimension_table(
    D: np.ndarray,
    dimensions: Iterable[int],
    *,
    metric_mds: bool,
    n_init: int,
    max_iter: int,
    eps: float,
    n_jobs: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[int, tuple[MDS, np.ndarray]]]:
    """Fit requested dimensions and build a stress/distance-preservation table."""

    unique_dimensions = sorted(set(int(d) for d in dimensions))
    if not unique_dimensions:
        raise ValueError("At least one MDS dimension must be requested")

    rows: list[dict[str, object]] = []
    fits: dict[int, tuple[MDS, np.ndarray]] = {}

    for n_components in unique_dimensions:
        model, embedding, stress1 = fit_mds(
            D,
            n_components=n_components,
            metric_mds=metric_mds,
            n_init=n_init,
            max_iter=max_iter,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        preservation = distance_preservation(D, embedding)
        fits[n_components] = (model, embedding)
        rows.append(
            {
                "n_components": n_components,
                "stress1": stress1,
                "n_iter": int(getattr(model, "n_iter_", -1)),
                **preservation,
            }
        )

    return pd.DataFrame(rows), fits


def _is_numeric_metadata(series: pd.Series) -> bool:
    """Treat only true numeric dtypes as continuous metadata."""

    return pd.api.types.is_numeric_dtype(series)


def plot_embedding(
    data: pd.DataFrame,
    output_path: Path,
    color_by: str | None = None,
    metric_mds: bool = True,
) -> None:
    """Save the two-dimensional MDS embedding."""

    fig, ax = plt.subplots(figsize=(7, 6))

    if color_by is None:
        ax.scatter(data["MDS1"], data["MDS2"], s=28, alpha=0.8)
    else:
        if color_by not in data.columns:
            raise KeyError(f"Metadata column not found: {color_by}")

        series = data[color_by]
        valid = series.notna()

        if _is_numeric_metadata(series):
            numeric = pd.to_numeric(series, errors="coerce")
            scatter = ax.scatter(
                data.loc[valid, "MDS1"],
                data.loc[valid, "MDS2"],
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
                    data.loc[mask, "MDS1"],
                    data.loc[mask, "MDS2"],
                    s=28,
                    alpha=0.8,
                    label=category,
                )

        missing = ~valid
        if missing.any():
            ax.scatter(
                data.loc[missing, "MDS1"],
                data.loc[missing, "MDS2"],
                s=28,
                alpha=0.45,
                marker="x",
                label="Missing" if not _is_numeric_metadata(series) else None,
            )

        if not _is_numeric_metadata(series):
            if valid.any() or missing.any():
                ax.legend(
                    title=color_by,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                    fontsize=8,
                )

    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    mds_name = "Metric MDS" if metric_mds else "Non-metric MDS"
    title = f"{mds_name} embedding" if color_by is None else f"{mds_name} — {color_by}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stress_curve(dimension_table: pd.DataFrame, output_path: Path) -> None:
    """Save normalized Stress-1 versus embedding dimensionality."""

    table = dimension_table.dropna(subset=["stress1"]).sort_values("n_components")
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["n_components"], table["stress1"], marker="o")
    ax.set_xlabel("MDS dimensions")
    ax.set_ylabel("Normalized stress (Stress-1)")
    ax.set_title("MDS stress by dimensionality")
    ax.set_xticks(table["n_components"].astype(int))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_shepard(
    D: np.ndarray,
    embedding: np.ndarray,
    output_path: Path,
    max_pairs: int,
    random_state: int,
) -> None:
    """Save a Shepard plot of original versus embedded distances."""

    original = squareform(D, checks=False)
    embedded = pdist(embedding, metric="euclidean")

    if max_pairs > 0 and len(original) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(original), size=max_pairs, replace=False)
        original_plot = original[idx]
        embedded_plot = embedded[idx]
    else:
        original_plot = original
        embedded_plot = embedded

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(original_plot, embedded_plot, s=10, alpha=0.3)
    ax.set_xlabel("Original pairwise dissimilarity")
    ax.set_ylabel("Distance in MDS embedding")
    ax.set_title("MDS Shepard plot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results(
    output_dir: str | Path,
    X: pd.DataFrame,
    sample_info: pd.DataFrame,
    D: np.ndarray,
    dimension_table: pd.DataFrame,
    fits: dict[int, tuple[MDS, np.ndarray]],
    *,
    primary_dimensions: int,
    metric_mds: bool,
    distance_metric: str,
    n_init: int,
    max_iter: int,
    eps: float,
    random_state: int,
    color_by: Iterable[str],
    shepard_max_pairs: int,
) -> dict[str, object]:
    """Save MDS coordinates, diagnostics, plots, and a JSON summary."""

    if primary_dimensions not in fits:
        raise ValueError(
            f"Primary dimensionality {primary_dimensions} was not fitted."
        )

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    primary_model, primary_embedding = fits[primary_dimensions]
    columns = [f"MDS{i}" for i in range(1, primary_dimensions + 1)]
    scores = pd.DataFrame(primary_embedding, index=X.index, columns=columns)
    scores.index.name = "sample_id"
    scores.to_csv(output_dir / "mds_scores.tsv.gz", sep="\t", compression="gzip")

    scores_with_metadata = (
        scores.reset_index()
        .merge(sample_info, on="sample_id", how="left", validate="one_to_one", sort=False)
    )
    scores_with_metadata.to_csv(
        output_dir / "mds_scores_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    dimension_table.to_csv(output_dir / "mds_dimension_diagnostics.tsv", sep="\t", index=False)

    # Saving the distance matrix makes the MDS analysis reproducible and lets us
    # reuse exactly the same sample dissimilarities for sensitivity analyses.
    np.save(output_dir / "mds_dissimilarity_matrix.npy", D)

    if primary_dimensions >= 2:
        plot_embedding(
            scores_with_metadata,
            figure_dir / "mds1_mds2.png",
            metric_mds=metric_mds,
        )

        requested_metadata = list(dict.fromkeys(color_by))
        for metadata_column in requested_metadata:
            if metadata_column not in scores_with_metadata.columns:
                available = ", ".join(sample_info.columns)
                raise KeyError(
                    f"Cannot color MDS by {metadata_column!r}; column not found. "
                    f"Available metadata columns: {available}"
                )
            safe_name = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_"
                for ch in metadata_column
            )
            plot_embedding(
                scores_with_metadata,
                figure_dir / f"mds1_mds2_by_{safe_name}.png",
                color_by=metadata_column,
                metric_mds=metric_mds,
            )
    else:
        requested_metadata = list(dict.fromkeys(color_by))

    plot_stress_curve(dimension_table, figure_dir / "mds_stress_by_dimension.png")
    plot_shepard(
        D,
        primary_embedding,
        figure_dir / "mds_shepard.png",
        max_pairs=shepard_max_pairs,
        random_state=random_state,
    )

    primary_row = dimension_table.loc[
        dimension_table["n_components"].eq(primary_dimensions)
    ].iloc[0]

    summary: dict[str, object] = {
        "input": {
            "samples": int(X.shape[0]),
            "genes": int(X.shape[1]),
        },
        "mds": {
            "metric_mds": bool(metric_mds),
            "distance_metric": distance_metric,
            "primary_dimensions": int(primary_dimensions),
            "dimension_grid": [int(x) for x in dimension_table["n_components"]],
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "eps": float(eps),
            "random_state": int(random_state),
            "primary_stress1": (
                None if pd.isna(primary_row["stress1"]) else float(primary_row["stress1"])
            ),
            "primary_n_iter": int(primary_row["n_iter"]),
            "primary_pearson_r": float(primary_row["pearson_r"]),
            "primary_spearman_rho": float(primary_row["spearman_rho"]),
        },
        "outputs": {
            "metadata_overlays": requested_metadata,
        },
    }

    with (output_dir / "mds_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def format_summary(summary: dict[str, object]) -> str:
    """Format MDS results for terminal output."""

    input_info = summary["input"]
    mds_info = summary["mds"]
    stress = mds_info["primary_stress1"]
    stress_text = "unavailable" if stress is None else f"{stress:.4f}"

    return "\n".join(
        [
            "TCGA STAD MDS summary",
            "---------------------",
            f"Samples:                         {input_info['samples']:,}",
            f"Genes:                           {input_info['genes']:,}",
            f"MDS type:                        {'metric' if mds_info['metric_mds'] else 'non-metric'}",
            f"Input distance metric:           {mds_info['distance_metric']}",
            f"Primary embedding dimensions:    {mds_info['primary_dimensions']}",
            f"Normalized stress (Stress-1):    {stress_text}",
            f"Iterations:                      {mds_info['primary_n_iter']}",
            f"Pairwise-distance Pearson r:     {mds_info['primary_pearson_r']:.4f}",
            f"Pairwise-distance Spearman rho:  {mds_info['primary_spearman_rho']:.4f}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MDS on the canonical TCGA STAD dimensionality-reduction matrix."
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Directory created by preprocessing.py (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/mds",
        help="Directory for MDS outputs (default: analysis/mds)",
    )
    parser.add_argument(
        "--distance-metric",
        default="euclidean",
        help=(
            "Sample-to-sample distance metric accepted by scipy.spatial.distance.pdist "
            "(default: euclidean). Correlation and cosine are useful sensitivity analyses."
        ),
    )
    parser.add_argument(
        "--nonmetric",
        action="store_true",
        help="Use non-metric MDS instead of the default metric MDS.",
    )
    parser.add_argument(
        "--primary-dimensions",
        type=int,
        default=2,
        help="Embedding dimensionality saved as the primary MDS result (default: 2)",
    )
    parser.add_argument(
        "--dimension-grid",
        type=int,
        nargs="*",
        default=list(DEFAULT_DIMENSION_GRID),
        help="Dimensions evaluated for stress diagnostics (default: 2 3 5)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=4,
        help="Independent SMACOF initializations per dimension (default: 4)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum SMACOF iterations per initialization (default: 300)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help=(
            "SMACOF convergence tolerance (default: 1e-3, matching the historical "
            "project/reference configuration rather than newer sklearn defaults)"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs across MDS initializations (default: -1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible MDS initialization (default: 42)",
    )
    parser.add_argument(
        "--color-by",
        nargs="*",
        default=list(DEFAULT_COLOR_BY),
        help=(
            "Metadata columns used for MDS1/MDS2 overlays. Default: center_code. "
            "Example: --color-by center_code gender race histological_grade"
        ),
    )
    parser.add_argument(
        "--shepard-max-pairs",
        type=int,
        default=25000,
        help="Maximum random sample of point pairs plotted in Shepard plot (default: 25000)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.primary_dimensions < 1:
        raise ValueError("--primary-dimensions must be at least 1")
    if args.n_init < 1:
        raise ValueError("--n-init must be at least 1")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")
    if args.shepard_max_pairs < 0:
        raise ValueError("--shepard-max-pairs cannot be negative")

    dimensions = list(args.dimension_grid)
    if args.primary_dimensions not in dimensions:
        dimensions.append(args.primary_dimensions)

    X, sample_info = load_inputs(args.processed_dir)
    D = compute_dissimilarities(X, metric=args.distance_metric)

    dimension_table, fits = build_dimension_table(
        D,
        dimensions,
        metric_mds=not args.nonmetric,
        n_init=args.n_init,
        max_iter=args.max_iter,
        eps=args.eps,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    summary = save_results(
        output_dir=args.output_dir,
        X=X,
        sample_info=sample_info,
        D=D,
        dimension_table=dimension_table,
        fits=fits,
        primary_dimensions=args.primary_dimensions,
        metric_mds=not args.nonmetric,
        distance_metric=args.distance_metric,
        n_init=args.n_init,
        max_iter=args.max_iter,
        eps=args.eps,
        random_state=args.random_state,
        color_by=args.color_by,
        shepard_max_pairs=args.shepard_max_pairs,
    )

    print(format_summary(summary))
    print("\nDimension diagnostics:")
    print(dimension_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()