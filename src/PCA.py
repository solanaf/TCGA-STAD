#!/usr/bin/env python3
"""Principal component analysis for the rebuilt TCGA STAD project.

This script consumes the canonical dimensionality-reduction matrix produced by
``preprocessing.py`` and performs PCA without any additional filtering or
scaling. It saves sample scores, explained-variance statistics, gene loadings,
and a small set of diagnostic plots.

Expected inputs in ``--processed-dir``
--------------------------------------
X_dimred.tsv.gz
    Samples x genes matrix. Rows are indexed by ``sample_id`` and columns are
    genes. The canonical preprocessing pipeline has already log-transformed,
    variance-filtered, and gene-wise z-scored this matrix.
sample_info.tsv
    Sample identifiers plus patient/technical/clinical metadata.
gene_info.tsv
    Gene annotation and preprocessing audit fields.

The script intentionally does *not* perform clustering or formal metadata
association testing. Those are kept separate so that PCA, MDS, t-SNE, and UMAP
can later be compared using the same downstream evaluation workflow.

Run directly with:

python pca.py \
    --processed-dir processed \
    --output-dir analysis/pca \
    --color-by \
        center_code \
        gender \
        race \
        ajcc_pathologic_tumor_stage \
        histological_grade \
        vital_status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_VARIANCE_THRESHOLDS = (0.50, 0.80, 0.90, 0.95)
DEFAULT_COLOR_BY = ("center_code",)


def load_inputs(
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate preprocessing outputs."""

    processed_dir = Path(processed_dir)
    x_path = processed_dir / "X_dimred.tsv.gz"
    sample_path = processed_dir / "sample_info.tsv"
    gene_path = processed_dir / "gene_info.tsv"

    missing = [p.name for p in (x_path, sample_path, gene_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required preprocessing output(s) in {processed_dir}: "
            + ", ".join(missing)
        )

    X = pd.read_csv(x_path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.index.name = "sample_id"

    # Keep barcode-derived fields as strings even when they look numeric.
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
    gene_info = pd.read_csv(gene_path, sep="\t", low_memory=False)

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

    # Match metadata row order to the expression matrix exactly.
    sample_info = (
        sample_info.set_index("sample_id")
        .loc[X.index]
        .reset_index()
    )

    values = X.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("X_dimred contains NaN or infinite values")

    return X, sample_info, gene_info


def fit_pca(X: pd.DataFrame) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA across the full informative sample-space rank.

    With centered data, at most ``n_samples - 1`` principal components carry
    non-zero variance. Fitting this full rank lets us construct an honest
    cumulative explained-variance curve and determine how many components are
    required to reach common variance thresholds.
    """

    n_samples, n_features = X.shape
    n_components = min(n_samples - 1, n_features)
    if n_components < 2:
        raise ValueError("PCA requires at least two informative dimensions")

    model = PCA(n_components=n_components, svd_solver="full")
    transformed = model.fit_transform(X)

    columns = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(transformed, index=X.index, columns=columns)
    scores.index.name = "sample_id"
    return model, scores


def build_variance_table(model: PCA) -> pd.DataFrame:
    """Create per-component explained-variance statistics."""

    ratios = model.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(ratios) + 1)],
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": ratios,
            "explained_variance_percent": ratios * 100.0,
            "cumulative_explained_variance_ratio": np.cumsum(ratios),
            "cumulative_explained_variance_percent": np.cumsum(ratios) * 100.0,
        }
    )


def components_for_thresholds(
    variance_table: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_VARIANCE_THRESHOLDS,
) -> dict[str, int | None]:
    """Return the smallest component count reaching each variance threshold."""

    cumulative = variance_table["cumulative_explained_variance_ratio"].to_numpy()
    result: dict[str, int | None] = {}

    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError(f"Variance threshold must be in (0, 1]: {threshold}")
        idx = np.searchsorted(cumulative, threshold, side="left")
        result[f"{int(round(threshold * 100))}_percent"] = (
            int(idx + 1) if idx < len(cumulative) else None
        )

    return result


def build_loadings(
    model: PCA,
    gene_columns: pd.Index,
    n_components: int,
) -> pd.DataFrame:
    """Return gene loadings for the requested leading PCs."""

    n_components = min(n_components, model.components_.shape[0])
    columns = [f"PC{i}" for i in range(1, n_components + 1)]
    loadings = pd.DataFrame(
        model.components_[:n_components].T,
        index=gene_columns,
        columns=columns,
    )
    loadings.index.name = "gene_column"
    return loadings


def build_top_loadings(
    loadings: pd.DataFrame,
    gene_info: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Collect the strongest positive and negative gene loadings per PC."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    annotation_columns = [
        c for c in ("gene_column", "gene_symbol", "entrez_id") if c in gene_info.columns
    ]
    annotations = (
        gene_info[annotation_columns]
        .drop_duplicates(subset="gene_column")
        .set_index("gene_column")
        if "gene_column" in annotation_columns
        else pd.DataFrame(index=loadings.index)
    )

    rows: list[dict[str, object]] = []
    for component in loadings.columns:
        s = loadings[component]
        selections = (
            ("positive", s.nlargest(top_n)),
            ("negative", s.nsmallest(top_n)),
        )
        for direction, selected in selections:
            for rank, (gene_column, loading) in enumerate(selected.items(), start=1):
                row: dict[str, object] = {
                    "component": component,
                    "direction": direction,
                    "rank": rank,
                    "gene_column": gene_column,
                    "loading": float(loading),
                    "absolute_loading": float(abs(loading)),
                }
                if gene_column in annotations.index:
                    for col in annotations.columns:
                        row[col] = annotations.at[gene_column, col]
                rows.append(row)

    return pd.DataFrame(rows)


def _component_percent(variance_table: pd.DataFrame, component: str) -> float:
    row = variance_table.loc[variance_table["component"] == component]
    if row.empty:
        raise KeyError(component)
    return float(row["explained_variance_percent"].iloc[0])


def plot_scree(
    variance_table: pd.DataFrame,
    output_path: Path,
    n_components: int,
) -> None:
    """Save a scree plot for the leading principal components."""

    n = min(n_components, len(variance_table))
    x = np.arange(1, n + 1)
    y = variance_table["explained_variance_percent"].iloc[:n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot")
    ax.set_xticks(x if n <= 20 else np.arange(1, n + 1, 2))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_variance(
    variance_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the cumulative explained-variance curve."""

    x = np.arange(1, len(variance_table) + 1)
    y = variance_table["cumulative_explained_variance_percent"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.axhline(80, linestyle="--", linewidth=1)
    ax.axhline(90, linestyle="--", linewidth=1)
    ax.axhline(95, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 100.5)
    ax.set_title("PCA cumulative explained variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _is_numeric_metadata(series: pd.Series) -> bool:
    """Return True only for metadata stored with a numeric dtype.

    Barcode-derived variables such as ``center_code`` intentionally remain
    strings even when all of their values contain digits, because they are
    categorical technical labels rather than continuous measurements.
    """

    return pd.api.types.is_numeric_dtype(series)


def plot_pc_pair(
    data: pd.DataFrame,
    variance_table: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    output_path: Path,
    color_by: str | None = None,
) -> None:
    """Save a scatterplot for a pair of principal components."""

    x_percent = _component_percent(variance_table, pc_x)
    y_percent = _component_percent(variance_table, pc_y)

    fig, ax = plt.subplots(figsize=(7, 6))

    if color_by is None:
        ax.scatter(data[pc_x], data[pc_y], s=28, alpha=0.8)
    else:
        if color_by not in data.columns:
            raise KeyError(f"Metadata column not found: {color_by}")

        series = data[color_by]
        valid = series.notna()

        if _is_numeric_metadata(series):
            numeric = pd.to_numeric(series, errors="coerce")
            scatter = ax.scatter(
                data.loc[valid, pc_x],
                data.loc[valid, pc_y],
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
                    data.loc[mask, pc_x],
                    data.loc[mask, pc_y],
                    s=28,
                    alpha=0.8,
                    label=category,
                )

        # Missing metadata remain visible but visually separate from categories.
        missing = ~valid
        if missing.any():
            ax.scatter(
                data.loc[missing, pc_x],
                data.loc[missing, pc_y],
                s=28,
                alpha=0.45,
                marker="x",
                label="Missing" if not _is_numeric_metadata(series) else None,
            )

        if not _is_numeric_metadata(series):
            categories = sorted(series.loc[valid].astype(str).unique())
            if categories or missing.any():
                ax.legend(
                    title=color_by,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                    fontsize=8,
                )

    ax.set_xlabel(f"{pc_x} ({x_percent:.2f}% variance)")
    ax.set_ylabel(f"{pc_y} ({y_percent:.2f}% variance)")
    title = f"{pc_x} vs {pc_y}"
    if color_by is not None:
        title += f" — {color_by}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_results(
    output_dir: str | Path,
    X: pd.DataFrame,
    sample_info: pd.DataFrame,
    gene_info: pd.DataFrame,
    model: PCA,
    scores: pd.DataFrame,
    variance_table: pd.DataFrame,
    save_loadings_components: int,
    top_loadings: int,
    scree_components: int,
    color_by: Iterable[str],
) -> dict[str, object]:
    """Save PCA tables, plots, and summary statistics."""

    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(output_dir / "pca_scores.tsv.gz", sep="\t", compression="gzip")

    scores_with_metadata = (
        scores.reset_index()
        .merge(sample_info, on="sample_id", how="left", validate="one_to_one", sort=False)
    )
    scores_with_metadata.to_csv(
        output_dir / "pca_scores_with_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    variance_table.to_csv(output_dir / "pca_variance.tsv", sep="\t", index=False)

    loadings = build_loadings(model, X.columns, save_loadings_components)
    loadings.to_csv(
        output_dir / "pca_loadings.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    top = build_top_loadings(loadings, gene_info, top_loadings)
    top.to_csv(output_dir / "pca_top_loadings.tsv", sep="\t", index=False)

    plot_scree(
        variance_table,
        figure_dir / "pca_scree.png",
        n_components=scree_components,
    )
    plot_cumulative_variance(
        variance_table,
        figure_dir / "pca_cumulative_variance.png",
    )

    # Always save the three pairwise views of the first three PCs.
    for pc_x, pc_y in (("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")):
        plot_pc_pair(
            scores_with_metadata,
            variance_table,
            pc_x,
            pc_y,
            figure_dir / f"{pc_x.lower()}_{pc_y.lower()}.png",
        )

    requested_metadata = list(dict.fromkeys(color_by))
    for metadata_column in requested_metadata:
        if metadata_column not in scores_with_metadata.columns:
            available = ", ".join(sample_info.columns)
            raise KeyError(
                f"Cannot color PCA by {metadata_column!r}; column not found. "
                f"Available metadata columns: {available}"
            )
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in metadata_column
        )
        plot_pc_pair(
            scores_with_metadata,
            variance_table,
            "PC1",
            "PC2",
            figure_dir / f"pc1_pc2_by_{safe_name}.png",
            color_by=metadata_column,
        )

    thresholds = components_for_thresholds(variance_table)
    summary: dict[str, object] = {
        "input": {
            "samples": int(X.shape[0]),
            "genes": int(X.shape[1]),
        },
        "pca": {
            "components_fit": int(model.n_components_),
            "svd_solver": model.svd_solver,
            "pc1_explained_variance_percent": float(
                variance_table.loc[0, "explained_variance_percent"]
            ),
            "pc2_explained_variance_percent": float(
                variance_table.loc[1, "explained_variance_percent"]
            ),
            "pc1_pc2_cumulative_variance_percent": float(
                variance_table.loc[1, "cumulative_explained_variance_percent"]
            ),
            "pc1_pc2_pc3_cumulative_variance_percent": float(
                variance_table.loc[2, "cumulative_explained_variance_percent"]
            ),
            "components_for_variance_threshold": thresholds,
        },
        "outputs": {
            "loadings_components_saved": int(loadings.shape[1]),
            "top_loadings_per_direction": int(top_loadings),
            "metadata_overlays": requested_metadata,
        },
    }

    with (output_dir / "pca_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def format_summary(summary: dict[str, object]) -> str:
    """Format PCA results for terminal output."""

    input_info = summary["input"]
    pca_info = summary["pca"]
    thresholds = pca_info["components_for_variance_threshold"]

    return "\n".join(
        [
            "TCGA STAD PCA summary",
            "---------------------",
            f"Samples:                         {input_info['samples']:,}",
            f"Genes:                           {input_info['genes']:,}",
            f"Principal components fit:        {pca_info['components_fit']:,}",
            f"PC1 variance explained:          {pca_info['pc1_explained_variance_percent']:.2f}%",
            f"PC2 variance explained:          {pca_info['pc2_explained_variance_percent']:.2f}%",
            f"PC1 + PC2 cumulative:            {pca_info['pc1_pc2_cumulative_variance_percent']:.2f}%",
            f"PC1 + PC2 + PC3 cumulative:      {pca_info['pc1_pc2_pc3_cumulative_variance_percent']:.2f}%",
            f"Components for 50% variance:      {thresholds['50_percent']}",
            f"Components for 80% variance:      {thresholds['80_percent']}",
            f"Components for 90% variance:      {thresholds['90_percent']}",
            f"Components for 95% variance:      {thresholds['95_percent']}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PCA on the canonical TCGA STAD dimensionality-reduction matrix."
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Directory created by preprocessing.py (default: processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/pca",
        help="Directory for PCA outputs (default: analysis/pca)",
    )
    parser.add_argument(
        "--save-loadings-components",
        type=int,
        default=20,
        help="Number of leading PCs whose gene loadings are saved (default: 20)",
    )
    parser.add_argument(
        "--top-loadings",
        type=int,
        default=20,
        help="Top positive and negative gene loadings to report per saved PC (default: 20)",
    )
    parser.add_argument(
        "--scree-components",
        type=int,
        default=25,
        help="Number of leading PCs displayed in the scree plot (default: 25)",
    )
    parser.add_argument(
        "--color-by",
        nargs="*",
        default=list(DEFAULT_COLOR_BY),
        help=(
            "Metadata columns used for PC1/PC2 overlay plots. "
            "Default: center_code. Example: --color-by center_code gender race"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.save_loadings_components < 1:
        raise ValueError("--save-loadings-components must be at least 1")
    if args.top_loadings < 1:
        raise ValueError("--top-loadings must be at least 1")
    if args.scree_components < 1:
        raise ValueError("--scree-components must be at least 1")

    X, sample_info, gene_info = load_inputs(args.processed_dir)
    model, scores = fit_pca(X)
    variance_table = build_variance_table(model)

    summary = save_results(
        output_dir=args.output_dir,
        X=X,
        sample_info=sample_info,
        gene_info=gene_info,
        model=model,
        scores=scores,
        variance_table=variance_table,
        save_loadings_components=args.save_loadings_components,
        top_loadings=args.top_loadings,
        scree_components=args.scree_components,
        color_by=args.color_by,
    )

    print(format_summary(summary))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()