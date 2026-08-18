#!/usr/bin/env python3
"""Metadata association analysis for TCGA STAD K-means clusters.

This script consumes the *selected* K-means cluster assignments produced by
``clustering.py`` and asks a separate downstream question:

    Are the clusters associated with technical or clinical metadata?

The clustering solution is therefore fixed before metadata are examined. This
avoids using metadata to choose the number of clusters or to tune the
dimensionality-reduction methods.

Important scope
---------------
By default, this script analyzes only the canonical/best K-means solution from
each dimensionality-reduction method:

    PCA   -> best k selected from canonical PC1/PC2 embedding
    MDS   -> best k selected from canonical MDS1/MDS2 embedding
    t-SNE -> best k selected from canonical tSNE1/tSNE2 embedding
    UMAP  -> best k selected from canonical UMAP1/UMAP2 embedding

It does NOT automatically analyze the t-SNE/UMAP parameter-sweep embeddings.
Those sweeps are sensitivity analyses performed in the dimensionality-reduction
scripts. If configuration-level clustering is added later, its cluster
assignments can be analyzed with the same statistical ideas used here.

Statistical tests
-----------------
Categorical metadata
    * 2 x 2 tables: Fisher's exact test.
    * Larger tables with adequate expected counts: Pearson chi-square test.
    * Larger sparse tables: permutation chi-square test.
    * Effect size: bias-corrected Cramer's V.

Continuous metadata
    * Kruskal-Wallis H test across clusters.
    * Effect size: epsilon-squared.

Multiple testing
    Benjamini-Hochberg FDR correction is applied:
      1. across all valid tests globally (``q_value``), and
      2. separately within categorical/continuous families
         (``q_value_family``).

Survival-time note
------------------
Time-to-event columns such as OS.time are deliberately NOT included as default
continuous variables because censoring makes an ordinary Kruskal-Wallis test
inappropriate for a primary survival analysis. Status variables (for example
OS or vital_status) may be explored categorically here. A later survival
analysis should use event + time together (for example Kaplan-Meier/log-rank or
a Cox model).

Expected input
--------------
analysis/clustering/
    best_k_summary.tsv
    best_cluster_assignments_with_metadata.tsv.gz

Outputs
-------
analysis/metadata_association/
    categorical_associations.tsv
    categorical_enrichment_long.tsv.gz
    continuous_associations.tsv
    continuous_group_summary.tsv
    all_associations.tsv
    metadata_availability.tsv
    metadata_association_summary.json
    figures/
        categorical_cramers_v_heatmap.png
        continuous_epsilon_squared_heatmap.png

Run directly with:

python metadata_association.py \
    --clustering-dir analysis/clustering \
    --output-dir analysis/metadata_association
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal


DEFAULT_CATEGORICAL = (
    "center_code",
    "gender",
    "race",
    "ajcc_pathologic_tumor_stage",
    "histological_type",
    "histological_grade",
    "tumor_status",
    "vital_status",
    "OS",
    "DSS",
    "DFI",
    "PFI",
)

DEFAULT_CONTINUOUS = (
    "age_at_initial_pathologic_diagnosis",
)

MISSING_STRINGS = {
    "",
    "nan",
    "none",
    "na",
    "n/a",
    "[not available]",
    "[not evaluated]",
    "[unknown]",
    "[discrepancy]",
}


def _safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def load_inputs(
    clustering_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected clusters + metadata and the best-k summary."""

    clustering_dir = Path(clustering_dir)
    assignment_path = (
        clustering_dir / "best_cluster_assignments_with_metadata.tsv.gz"
    )
    best_k_path = clustering_dir / "best_k_summary.tsv"

    missing = [p for p in (assignment_path, best_k_path) if not p.exists()]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required clustering output(s):\n"
            f"{formatted}\nRun clustering.py first."
        )

    data = pd.read_csv(
        assignment_path,
        sep="\t",
        dtype={
            "patient_id": "string",
            "sample_id": "string",
            "sample_type_code": "string",
            "tss_code": "string",
            "center_code": "string",
        },
    )
    best_k = pd.read_csv(best_k_path, sep="\t")

    if "sample_id" not in data.columns:
        raise ValueError(f"{assignment_path} must contain sample_id")
    if data["sample_id"].duplicated().any():
        raise ValueError(f"{assignment_path} contains duplicate sample_id values")

    required_best = {"method", "method_key", "k"}
    missing_best = required_best - set(best_k.columns)
    if missing_best:
        raise ValueError(
            f"{best_k_path} is missing required columns: {sorted(missing_best)}"
        )

    cluster_columns = [c for c in data.columns if c.endswith("_cluster")]
    if not cluster_columns:
        raise ValueError(
            f"No *_cluster columns found in {assignment_path}. "
            "Expected columns such as pca_cluster or umap_cluster."
        )

    return data, best_k


def normalize_missing(series: pd.Series) -> pd.Series:
    """Convert common placeholder strings to missing values."""

    if pd.api.types.is_numeric_dtype(series):
        return series

    out = series.copy()
    stringified = out.astype("string")
    mask = stringified.str.strip().str.lower().isin(MISSING_STRINGS)
    out = out.mask(mask)
    return out


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment preserving input order."""

    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)

    if not valid.any():
        return q

    valid_indices = np.flatnonzero(valid)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid_indices] = restored
    return q


def bias_corrected_cramers_v(
    chi2: float,
    n: int,
    n_rows: int,
    n_cols: int,
) -> float:
    """Bias-corrected Cramer's V for an r x c contingency table."""

    if n <= 1 or n_rows < 2 or n_cols < 2:
        return float("nan")

    phi2 = chi2 / n
    correction = ((n_cols - 1) * (n_rows - 1)) / (n - 1)
    phi2_corr = max(0.0, phi2 - correction)

    rows_corr = n_rows - ((n_rows - 1) ** 2) / (n - 1)
    cols_corr = n_cols - ((n_cols - 1) ** 2) / (n - 1)
    denominator = min(rows_corr - 1, cols_corr - 1)

    if denominator <= 0:
        return 0.0
    return float(math.sqrt(phi2_corr / denominator))


def _chi_square_stat_from_counts(
    observed: np.ndarray,
    expected: np.ndarray,
) -> float:
    mask = expected > 0
    return float(np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask]))


def permutation_chi_square_pvalue(
    cluster_codes: np.ndarray,
    category_codes: np.ndarray,
    *,
    n_rows: int,
    n_cols: int,
    observed_chi2: float,
    expected: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Permutation p-value for independence using a Pearson chi-square statistic.

    Cluster membership is held fixed while metadata labels are permuted across
    samples. Row and column marginals therefore remain unchanged, so the
    expected-count matrix is constant across permutations.
    """

    if n_permutations <= 0:
        raise ValueError("n_permutations must be > 0")

    ge_count = 0
    flat_size = n_rows * n_cols

    for _ in range(n_permutations):
        permuted = rng.permutation(category_codes)
        flat = cluster_codes * n_cols + permuted
        counts = np.bincount(flat, minlength=flat_size).reshape(n_rows, n_cols)
        statistic = _chi_square_stat_from_counts(counts, expected)
        if statistic >= observed_chi2 - 1e-12:
            ge_count += 1

    # +1 correction prevents an estimated p-value of exactly zero.
    return float((ge_count + 1) / (n_permutations + 1))


def categorical_association(
    clusters: pd.Series,
    metadata: pd.Series,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Test one cluster assignment against one categorical metadata variable."""

    frame = pd.DataFrame(
        {
            "cluster": clusters,
            "metadata": normalize_missing(metadata),
        }
    ).dropna()

    if frame.empty:
        return (
            {
                "status": "insufficient_data",
                "n": 0,
                "n_clusters": 0,
                "n_categories": 0,
            },
            pd.DataFrame(),
        )

    # Use string labels so numeric endpoints such as OS=0/1 are treated
    # categorically and serialize consistently.
    frame["cluster"] = frame["cluster"].astype(str)
    frame["metadata"] = frame["metadata"].astype(str)

    table = pd.crosstab(frame["cluster"], frame["metadata"], dropna=False)
    n_rows, n_cols = table.shape
    n = int(table.to_numpy().sum())

    if n_rows < 2 or n_cols < 2:
        return (
            {
                "status": "insufficient_levels",
                "n": n,
                "n_clusters": n_rows,
                "n_categories": n_cols,
            },
            pd.DataFrame(),
        )

    observed = table.to_numpy(dtype=float)
    chi2, chi_p, dof, expected = chi2_contingency(observed, correction=False)

    cells_lt5 = int((expected < 5).sum())
    frac_lt5 = float(cells_lt5 / expected.size)
    min_expected = float(expected.min())

    fisher_odds_ratio = float("nan")
    if table.shape == (2, 2):
        fisher_result = fisher_exact(observed)
        # scipy versions differ in the object returned by fisher_exact.
        if hasattr(fisher_result, "statistic"):
            fisher_odds_ratio = float(fisher_result.statistic)
            p_value = float(fisher_result.pvalue)
        else:
            fisher_odds_ratio = float(fisher_result[0])
            p_value = float(fisher_result[1])
        test = "fisher_exact"
        statistic = fisher_odds_ratio
    elif min_expected < 1 or frac_lt5 > 0.20:
        row_categories = pd.Categorical(frame["cluster"], categories=table.index)
        col_categories = pd.Categorical(frame["metadata"], categories=table.columns)
        p_value = permutation_chi_square_pvalue(
            row_categories.codes.astype(int),
            col_categories.codes.astype(int),
            n_rows=n_rows,
            n_cols=n_cols,
            observed_chi2=float(chi2),
            expected=expected,
            n_permutations=n_permutations,
            rng=rng,
        )
        test = f"permutation_chi_square_{n_permutations}"
        statistic = float(chi2)
    else:
        p_value = float(chi_p)
        test = "chi_square"
        statistic = float(chi2)

    cramers_v = bias_corrected_cramers_v(
        float(chi2), n=n, n_rows=n_rows, n_cols=n_cols
    )

    enrichment_rows: list[dict[str, object]] = []
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)

    for i, cluster_label in enumerate(table.index):
        for j, category_label in enumerate(table.columns):
            obs = float(observed[i, j])
            exp = float(expected[i, j])
            pearson_residual = (
                float((obs - exp) / math.sqrt(exp)) if exp > 0 else float("nan")
            )
            enrichment_rows.append(
                {
                    "cluster": cluster_label,
                    "category": category_label,
                    "observed": int(obs),
                    "expected": exp,
                    "observed_expected_ratio": (
                        float(obs / exp) if exp > 0 else float("nan")
                    ),
                    "pearson_residual": pearson_residual,
                    "within_cluster_percent": (
                        float(obs / row_totals[i] * 100.0)
                        if row_totals[i] > 0
                        else float("nan")
                    ),
                    "within_category_percent": (
                        float(obs / col_totals[j] * 100.0)
                        if col_totals[j] > 0
                        else float("nan")
                    ),
                }
            )

    return (
        {
            "status": "ok",
            "n": n,
            "n_clusters": n_rows,
            "n_categories": n_cols,
            "test": test,
            "statistic": statistic,
            "chi_square_statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": p_value,
            "cramers_v": cramers_v,
            "fisher_odds_ratio": fisher_odds_ratio,
            "min_expected_count": min_expected,
            "cells_expected_lt5": cells_lt5,
            "fraction_cells_expected_lt5": frac_lt5,
        },
        pd.DataFrame(enrichment_rows),
    )


def epsilon_squared_kruskal(h_statistic: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for a Kruskal-Wallis test."""

    if n <= k or k < 2:
        return float("nan")
    value = (h_statistic - k + 1) / (n - k)
    return float(max(0.0, value))


def continuous_association(
    clusters: pd.Series,
    metadata: pd.Series,
    *,
    min_group_size: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Kruskal-Wallis test of one continuous variable across clusters."""

    numeric = pd.to_numeric(normalize_missing(metadata), errors="coerce")
    frame = pd.DataFrame(
        {
            "cluster": clusters,
            "value": numeric,
        }
    ).dropna()

    if frame.empty:
        return (
            {
                "status": "insufficient_data",
                "n": 0,
                "n_clusters": 0,
            },
            pd.DataFrame(),
        )

    groups: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []

    for cluster_label, group in frame.groupby("cluster", sort=True):
        values = group["value"].to_numpy(dtype=float)
        if len(values) < min_group_size:
            continue

        groups.append(values)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summary_rows.append(
            {
                "cluster": cluster_label,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    k = len(groups)
    n = int(sum(len(v) for v in groups))
    if k < 2:
        return (
            {
                "status": "insufficient_groups",
                "n": n,
                "n_clusters": k,
            },
            pd.DataFrame(summary_rows),
        )

    result = kruskal(*groups, nan_policy="omit")
    h_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    effect = epsilon_squared_kruskal(h_statistic, n=n, k=k)

    return (
        {
            "status": "ok",
            "n": n,
            "n_clusters": k,
            "test": "kruskal_wallis",
            "statistic": h_statistic,
            "p_value": p_value,
            "epsilon_squared": effect,
        },
        pd.DataFrame(summary_rows),
    )


def metadata_availability_table(
    data: pd.DataFrame,
    variables: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_total = len(data)

    for variable in variables:
        if variable not in data.columns:
            rows.append(
                {
                    "variable": variable,
                    "present_in_file": False,
                    "n_total": n_total,
                    "n_available": 0,
                    "percent_available": 0.0,
                    "n_unique_nonmissing": 0,
                }
            )
            continue

        clean = normalize_missing(data[variable])
        n_available = int(clean.notna().sum())
        rows.append(
            {
                "variable": variable,
                "present_in_file": True,
                "n_total": n_total,
                "n_available": n_available,
                "percent_available": (
                    float(n_available / n_total * 100.0) if n_total else float("nan")
                ),
                "n_unique_nonmissing": int(clean.dropna().nunique()),
            }
        )

    return pd.DataFrame(rows)


def _method_mapping(best_k: pd.DataFrame) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for row in best_k.itertuples(index=False):
        mapping[str(row.method_key)] = {
            "method": str(row.method),
            "k": int(row.k),
        }
    return mapping


def apply_fdr(
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply global and within-family BH FDR corrections."""

    cat = categorical.copy()
    cont = continuous.copy()

    if not cat.empty:
        cat["q_value_family"] = benjamini_hochberg(cat["p_value"])
    else:
        cat["q_value_family"] = pd.Series(dtype=float)

    if not cont.empty:
        cont["q_value_family"] = benjamini_hochberg(cont["p_value"])
    else:
        cont["q_value_family"] = pd.Series(dtype=float)

    all_parts: list[pd.DataFrame] = []
    if not cat.empty:
        cat_all = cat.copy()
        cat_all["family"] = "categorical"
        cat_all["effect_size"] = cat_all["cramers_v"]
        cat_all["effect_size_name"] = "cramers_v"
        all_parts.append(cat_all)
    if not cont.empty:
        cont_all = cont.copy()
        cont_all["family"] = "continuous"
        cont_all["effect_size"] = cont_all["epsilon_squared"]
        cont_all["effect_size_name"] = "epsilon_squared"
        all_parts.append(cont_all)

    if not all_parts:
        return cat, cont, pd.DataFrame()

    all_assoc = pd.concat(all_parts, ignore_index=True, sort=False)
    all_assoc["q_value"] = benjamini_hochberg(all_assoc["p_value"])

    # Push global q-values back to family-specific tables by stable test_id.
    q_map = all_assoc.set_index("test_id")["q_value"]
    if not cat.empty:
        cat["q_value"] = cat["test_id"].map(q_map)
    if not cont.empty:
        cont["q_value"] = cont["test_id"].map(q_map)

    return cat, cont, all_assoc


def plot_effect_heatmap(
    associations: pd.DataFrame,
    *,
    effect_column: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot method x metadata effect sizes, annotating global q-values."""

    if associations.empty:
        return

    pivot = associations.pivot(
        index="method",
        columns="variable",
        values=effect_column,
    )
    q_pivot = associations.pivot(
        index="method",
        columns="variable",
        values="q_value",
    )

    if pivot.empty:
        return

    width = max(7.0, 1.15 * len(pivot.columns) + 2.5)
    height = max(3.5, 0.8 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))

    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            effect = pivot.iloc[i, j]
            q = q_pivot.iloc[i, j]
            if pd.isna(effect):
                text = ""
            elif pd.isna(q):
                text = f"{effect:.2f}"
            elif q < 0.001:
                text = f"{effect:.2f}\nq<.001"
            else:
                text = f"{effect:.2f}\nq={q:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label=effect_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_terminal_summary(
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
    *,
    alpha: float,
) -> str:
    lines = [
        "TCGA STAD metadata association summary",
        "--------------------------------------",
        "Clusters were fixed BEFORE metadata testing.",
        f"Global Benjamini-Hochberg FDR threshold: q < {alpha:g}",
        "",
    ]

    n_cat = int((categorical["status"] == "ok").sum()) if not categorical.empty else 0
    n_cont = int((continuous["status"] == "ok").sum()) if not continuous.empty else 0
    lines.append(f"Valid categorical tests: {n_cat}")
    lines.append(f"Valid continuous tests:  {n_cont}")

    significant_parts: list[pd.DataFrame] = []
    if not categorical.empty and "q_value" in categorical:
        tmp = categorical.loc[
            (categorical["status"] == "ok") & (categorical["q_value"] < alpha)
        ].copy()
        if not tmp.empty:
            tmp["effect"] = tmp["cramers_v"]
            tmp["effect_name"] = "Cramer's V"
            significant_parts.append(tmp)

    if not continuous.empty and "q_value" in continuous:
        tmp = continuous.loc[
            (continuous["status"] == "ok") & (continuous["q_value"] < alpha)
        ].copy()
        if not tmp.empty:
            tmp["effect"] = tmp["epsilon_squared"]
            tmp["effect_name"] = "epsilon^2"
            significant_parts.append(tmp)

    lines.append("")
    if not significant_parts:
        lines.append("No metadata associations passed the global FDR threshold.")
        return "\n".join(lines)

    sig = pd.concat(significant_parts, ignore_index=True, sort=False)
    sig = sig.sort_values(["q_value", "p_value"])

    lines.append("Associations passing global FDR:")
    lines.append(
        f"{'Method':<8} {'Variable':<36} {'Test':<28} "
        f"{'p':>10} {'q':>10} {'Effect':>10}"
    )
    for row in sig.itertuples(index=False):
        lines.append(
            f"{str(row.method):<8} "
            f"{str(row.variable)[:36]:<36} "
            f"{str(row.test)[:28]:<28} "
            f"{row.p_value:>10.3g} "
            f"{row.q_value:>10.3g} "
            f"{row.effect:>10.3f}"
        )

    return "\n".join(lines)


def run_metadata_association(
    *,
    clustering_dir: str | Path,
    output_dir: str | Path,
    categorical_variables: Sequence[str],
    continuous_variables: Sequence[str],
    n_permutations: int,
    random_state: int,
    min_group_size: int,
    fdr_alpha: float,
) -> dict[str, object]:
    clustering_dir = Path(clustering_dir)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, best_k = load_inputs(clustering_dir)
    method_map = _method_mapping(best_k)

    cluster_columns: list[tuple[str, str, int, str]] = []
    for key, info in method_map.items():
        column = f"{key}_cluster"
        if column in data.columns:
            cluster_columns.append(
                (key, str(info["method"]), int(info["k"]), column)
            )

    if not cluster_columns:
        raise ValueError(
            "None of the best-k methods have corresponding *_cluster columns "
            "in best_cluster_assignments_with_metadata.tsv.gz."
        )

    requested_variables = list(
        dict.fromkeys([*categorical_variables, *continuous_variables])
    )
    availability = metadata_availability_table(data, requested_variables)
    availability.to_csv(
        output_dir / "metadata_availability.tsv",
        sep="\t",
        index=False,
    )

    missing_requested = availability.loc[
        ~availability["present_in_file"], "variable"
    ].tolist()
    if missing_requested:
        print(
            "Warning: requested metadata columns not found and will be skipped: "
            + ", ".join(missing_requested)
        )

    rng = np.random.default_rng(random_state)

    categorical_rows: list[dict[str, object]] = []
    enrichment_parts: list[pd.DataFrame] = []

    for method_key, method_name, k, cluster_column in cluster_columns:
        for variable in categorical_variables:
            if variable not in data.columns:
                continue

            result, enrichment = categorical_association(
                data[cluster_column],
                data[variable],
                n_permutations=n_permutations,
                rng=rng,
            )
            test_id = f"categorical::{method_key}::{variable}"
            row = {
                "test_id": test_id,
                "method": method_name,
                "method_key": method_key,
                "k": k,
                "variable": variable,
                **result,
            }
            categorical_rows.append(row)

            if not enrichment.empty:
                enrichment.insert(0, "variable", variable)
                enrichment.insert(0, "k", k)
                enrichment.insert(0, "method_key", method_key)
                enrichment.insert(0, "method", method_name)
                enrichment_parts.append(enrichment)

    continuous_rows: list[dict[str, object]] = []
    continuous_summary_parts: list[pd.DataFrame] = []

    for method_key, method_name, k, cluster_column in cluster_columns:
        for variable in continuous_variables:
            if variable not in data.columns:
                continue

            result, group_summary = continuous_association(
                data[cluster_column],
                data[variable],
                min_group_size=min_group_size,
            )
            test_id = f"continuous::{method_key}::{variable}"
            row = {
                "test_id": test_id,
                "method": method_name,
                "method_key": method_key,
                "k": k,
                "variable": variable,
                **result,
            }
            continuous_rows.append(row)

            if not group_summary.empty:
                group_summary.insert(0, "variable", variable)
                group_summary.insert(0, "k", k)
                group_summary.insert(0, "method_key", method_key)
                group_summary.insert(0, "method", method_name)
                continuous_summary_parts.append(group_summary)

    categorical = pd.DataFrame(categorical_rows)
    continuous = pd.DataFrame(continuous_rows)

    # Ensure columns needed downstream exist even if all tests are untestable.
    for frame, columns in (
        (
            categorical,
            [
                "status",
                "p_value",
                "cramers_v",
                "test",
            ],
        ),
        (
            continuous,
            [
                "status",
                "p_value",
                "epsilon_squared",
                "test",
            ],
        ),
    ):
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan

    categorical, continuous, all_assoc = apply_fdr(categorical, continuous)

    categorical.to_csv(
        output_dir / "categorical_associations.tsv",
        sep="\t",
        index=False,
    )
    continuous.to_csv(
        output_dir / "continuous_associations.tsv",
        sep="\t",
        index=False,
    )
    all_assoc.to_csv(
        output_dir / "all_associations.tsv",
        sep="\t",
        index=False,
    )

    enrichment_long = (
        pd.concat(enrichment_parts, ignore_index=True)
        if enrichment_parts
        else pd.DataFrame()
    )
    enrichment_long.to_csv(
        output_dir / "categorical_enrichment_long.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    continuous_summary = (
        pd.concat(continuous_summary_parts, ignore_index=True)
        if continuous_summary_parts
        else pd.DataFrame()
    )
    continuous_summary.to_csv(
        output_dir / "continuous_group_summary.tsv",
        sep="\t",
        index=False,
    )

    valid_cat = categorical.loc[categorical["status"] == "ok"].copy()
    valid_cont = continuous.loc[continuous["status"] == "ok"].copy()

    plot_effect_heatmap(
        valid_cat,
        effect_column="cramers_v",
        output_path=figure_dir / "categorical_cramers_v_heatmap.png",
        title="Cluster association with categorical metadata",
    )
    plot_effect_heatmap(
        valid_cont,
        effect_column="epsilon_squared",
        output_path=figure_dir / "continuous_epsilon_squared_heatmap.png",
        title="Cluster association with continuous metadata",
    )

    summary: dict[str, object] = {
        "input": {
            "samples": int(len(data)),
            "clustering_dir": str(clustering_dir),
            "methods": {
                key: {
                    "name": method,
                    "best_k": k,
                    "cluster_column": column,
                }
                for key, method, k, column in cluster_columns
            },
        },
        "categorical": {
            "requested_variables": list(categorical_variables),
            "valid_tests": int((categorical["status"] == "ok").sum())
            if not categorical.empty
            else 0,
            "n_permutations_for_sparse_tables": int(n_permutations),
        },
        "continuous": {
            "requested_variables": list(continuous_variables),
            "valid_tests": int((continuous["status"] == "ok").sum())
            if not continuous.empty
            else 0,
            "min_group_size": int(min_group_size),
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg",
            "global_fdr_alpha": float(fdr_alpha),
            "global_tests": int(all_assoc["p_value"].notna().sum())
            if not all_assoc.empty
            else 0,
            "significant_global_fdr": int(
                (all_assoc["q_value"] < fdr_alpha).sum()
            )
            if not all_assoc.empty
            else 0,
        },
        "notes": {
            "cluster_selection_used_metadata": False,
            "parameter_sweep_embeddings_analyzed": False,
            "survival_time_columns_defaulted_to_continuous": False,
        },
    }

    with (output_dir / "metadata_association_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(
        format_terminal_summary(
            categorical,
            continuous,
            alpha=fdr_alpha,
        )
    )
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the selected PCA/MDS/t-SNE/UMAP K-means clusters for "
            "association with technical and clinical metadata."
        )
    )
    parser.add_argument(
        "--clustering-dir",
        default="analysis/clustering",
        help=(
            "Directory containing clustering.py outputs "
            "(default: analysis/clustering)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/metadata_association",
        help="Output directory (default: analysis/metadata_association)",
    )
    parser.add_argument(
        "--categorical",
        nargs="*",
        default=list(DEFAULT_CATEGORICAL),
        help=(
            "Categorical metadata variables to test. Pass the flag with no "
            "values to disable categorical tests."
        ),
    )
    parser.add_argument(
        "--continuous",
        nargs="*",
        default=list(DEFAULT_CONTINUOUS),
        help=(
            "Continuous metadata variables to test. Pass the flag with no "
            "values to disable continuous tests."
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help=(
            "Permutations for sparse categorical tables larger than 2x2 "
            "(default: 5000)"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for permutation tests (default: 42)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help=(
            "Minimum nonmissing observations required for a cluster to enter "
            "a continuous-variable test (default: 2)"
        ),
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="Global Benjamini-Hochberg FDR threshold (default: 0.05)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be >= 1")
    if args.min_group_size < 1:
        raise ValueError("--min-group-size must be >= 1")
    if not 0 < args.fdr_alpha < 1:
        raise ValueError("--fdr-alpha must be between 0 and 1")

    run_metadata_association(
        clustering_dir=args.clustering_dir,
        output_dir=args.output_dir,
        categorical_variables=args.categorical,
        continuous_variables=args.continuous,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        min_group_size=args.min_group_size,
        fdr_alpha=args.fdr_alpha,
    )


if __name__ == "__main__":
    main()