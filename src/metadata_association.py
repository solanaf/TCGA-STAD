#!/usr/bin/env python3
"""Metadata association analysis for TCGA STAD K-means clusters.

This script consumes the selected best-k cluster solution for *every cached
embedding configuration* produced by ``clustering.py`` and asks:

    Are those fixed clusters associated with technical or clinical metadata?

The dimensionality-reduction configuration and K-means k are fixed before
metadata are examined. Metadata are therefore never used to tune t-SNE/UMAP
metrics, perplexity, n_neighbors, min_dist, or k.

Primary vs sensitivity analysis
-------------------------------
Canonical PCA, MDS, t-SNE, and UMAP configurations are marked ``is_primary``.
All other cached t-SNE/UMAP configurations are treated as sensitivity analyses.
The output tables contain both, with explicit ``configuration_id`` and parameter
columns so a later Streamlit app can switch between configurations without
rerunning statistics.

Multiple-testing correction is reported at several scopes:

``q_value``
    BH FDR across every valid test from every configuration.
``q_value_family``
    BH FDR within categorical or continuous tests across all configurations.
``q_value_scope``
    BH FDR separately within the primary and sensitivity analysis scopes.
    This is the preferred q-value for interpreting the canonical primary
    analysis without allowing the exploratory parameter sweep to change its
    correction burden.
``q_value_family_scope``
    BH FDR within family and analysis scope.

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

Time-to-event columns remain excluded from default continuous testing because
censoring requires survival-specific methods.

Expected input
--------------
analysis/clustering/
    best_k_summary_all.tsv
    best_cluster_assignments_all_with_metadata.tsv.gz

Outputs
-------
analysis/metadata_association/
    categorical_associations.tsv
    categorical_enrichment_long.tsv.gz
    continuous_associations.tsv
    continuous_group_summary.tsv
    all_associations.tsv
    primary_all_associations.tsv
    sensitivity_all_associations.tsv
    metadata_availability.tsv
    configuration_manifest.tsv
    metadata_association_summary.json
    figures/
        categorical_cramers_v_heatmap.png
        continuous_epsilon_squared_heatmap.png

The static heatmaps intentionally show the primary configurations only; all
sensitivity results remain cached in the long tables for interactive display.

Run directly with:
python metadata_association.py \
    --clustering-dir analysis/clustering \
    --output-dir analysis/metadata_association

add --primary-only to only execute canonical analyses
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_CONTINUOUS = ("age_at_initial_pathologic_diagnosis",)

CONFIG_COLUMNS = (
    "method",
    "method_key",
    "configuration_id",
    "configuration_label",
    "is_primary",
    "k",
    "metric",
    "perplexity",
    "n_neighbors",
    "min_dist",
    "pre_pca_components",
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


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_inputs(
    clustering_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all selected configuration-level clusters + metadata."""

    clustering_dir = Path(clustering_dir)
    assignment_path = clustering_dir / "best_cluster_assignments_all_with_metadata.tsv.gz"
    best_k_path = clustering_dir / "best_k_summary_all.tsv"

    missing = [p for p in (assignment_path, best_k_path) if not p.exists()]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing configuration-level clustering output(s):\n"
            f"{formatted}\nRun the updated clustering.py first."
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
            "configuration_id": "string",
        },
    )
    best_k = pd.read_csv(best_k_path, sep="\t")

    required_data = {"sample_id", "configuration_id", "cluster", "method_key", "is_primary"}
    missing_data = required_data - set(data.columns)
    if missing_data:
        raise ValueError(f"{assignment_path} is missing columns: {sorted(missing_data)}")

    required_best = {"method", "method_key", "configuration_id", "is_primary", "k"}
    missing_best = required_best - set(best_k.columns)
    if missing_best:
        raise ValueError(f"{best_k_path} is missing columns: {sorted(missing_best)}")

    data["sample_id"] = data["sample_id"].astype(str)
    data["configuration_id"] = data["configuration_id"].astype(str)
    best_k["configuration_id"] = best_k["configuration_id"].astype(str)
    data["is_primary"] = data["is_primary"].map(_truthy)
    best_k["is_primary"] = best_k["is_primary"].map(_truthy)

    if data.duplicated(["configuration_id", "sample_id"]).any():
        raise ValueError(
            f"{assignment_path} contains duplicate configuration_id/sample_id rows"
        )
    if best_k["configuration_id"].duplicated().any():
        raise ValueError(f"{best_k_path} contains duplicate configuration_id values")

    data_configs = set(data["configuration_id"])
    best_configs = set(best_k["configuration_id"])
    if data_configs != best_configs:
        raise ValueError(
            "Configuration mismatch between selected assignments and best-k summary. "
            f"Missing assignments={sorted(best_configs - data_configs)[:5]}; "
            f"extra assignments={sorted(data_configs - best_configs)[:5]}"
        )

    return data, best_k


def normalize_missing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    out = series.copy()
    stringified = out.astype("string")
    mask = stringified.str.strip().str.lower().isin(MISSING_STRINGS)
    return out.mask(mask)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
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


def bias_corrected_cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
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


def _chi_square_stat_from_counts(observed: np.ndarray, expected: np.ndarray) -> float:
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
    return float((ge_count + 1) / (n_permutations + 1))


def categorical_association(
    clusters: pd.Series,
    metadata: pd.Series,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    frame = pd.DataFrame(
        {"cluster": clusters, "metadata": normalize_missing(metadata)}
    ).dropna()

    if frame.empty:
        return ({"status": "insufficient_data", "n": 0, "n_clusters": 0, "n_categories": 0}, pd.DataFrame())

    frame["cluster"] = frame["cluster"].astype(str)
    frame["metadata"] = frame["metadata"].astype(str)
    table = pd.crosstab(frame["cluster"], frame["metadata"], dropna=False)
    n_rows, n_cols = table.shape
    n = int(table.to_numpy().sum())

    if n_rows < 2 or n_cols < 2:
        return (
            {"status": "insufficient_levels", "n": n, "n_clusters": n_rows, "n_categories": n_cols},
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

    cramers_v = bias_corrected_cramers_v(float(chi2), n=n, n_rows=n_rows, n_cols=n_cols)
    enrichment_rows: list[dict[str, object]] = []
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)

    for i, cluster_label in enumerate(table.index):
        for j, category_label in enumerate(table.columns):
            obs = float(observed[i, j])
            exp = float(expected[i, j])
            enrichment_rows.append(
                {
                    "cluster": cluster_label,
                    "category": category_label,
                    "observed": int(obs),
                    "expected": exp,
                    "observed_expected_ratio": float(obs / exp) if exp > 0 else float("nan"),
                    "pearson_residual": float((obs - exp) / math.sqrt(exp)) if exp > 0 else float("nan"),
                    "within_cluster_percent": float(obs / row_totals[i] * 100.0) if row_totals[i] > 0 else float("nan"),
                    "within_category_percent": float(obs / col_totals[j] * 100.0) if col_totals[j] > 0 else float("nan"),
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
    if n <= k or k < 2:
        return float("nan")
    return float(max(0.0, (h_statistic - k + 1) / (n - k)))


def continuous_association(
    clusters: pd.Series,
    metadata: pd.Series,
    *,
    min_group_size: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    numeric = pd.to_numeric(normalize_missing(metadata), errors="coerce")
    frame = pd.DataFrame({"cluster": clusters, "value": numeric}).dropna()
    if frame.empty:
        return ({"status": "insufficient_data", "n": 0, "n_clusters": 0}, pd.DataFrame())

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
        return ({"status": "insufficient_groups", "n": n, "n_clusters": k}, pd.DataFrame(summary_rows))

    result = kruskal(*groups, nan_policy="omit")
    h_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    return (
        {
            "status": "ok",
            "n": n,
            "n_clusters": k,
            "test": "kruskal_wallis",
            "statistic": h_statistic,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared_kruskal(h_statistic, n=n, k=k),
        },
        pd.DataFrame(summary_rows),
    )


def metadata_availability_table(data: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
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
                "percent_available": float(n_available / n_total * 100.0) if n_total else float("nan"),
                "n_unique_nonmissing": int(clean.dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def _config_metadata(row: pd.Series | object) -> dict[str, object]:
    if isinstance(row, pd.Series):
        getter = row.get
    else:
        getter = lambda key, default=None: getattr(row, key, default)
    out: dict[str, object] = {}
    for column in CONFIG_COLUMNS:
        value = getter(column, np.nan)
        if isinstance(value, np.generic):
            value = value.item()
        out[column] = value
    out["is_primary"] = _truthy(out["is_primary"])
    out["analysis_scope"] = "primary" if out["is_primary"] else "sensitivity"
    return out


def _rng_for_test(random_state: int, test_id: str) -> np.random.Generator:
    """Deterministic RNG per test so adding sensitivity tests cannot alter primary p-values."""

    digest = hashlib.sha256(f"{random_state}::{test_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    return np.random.default_rng(seed)


def apply_fdr(
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply BH corrections globally, by family, by scope, and family-within-scope."""

    cat = categorical.copy()
    cont = continuous.copy()
    parts: list[pd.DataFrame] = []
    if not cat.empty:
        tmp = cat.copy()
        tmp["family"] = "categorical"
        tmp["effect_size"] = tmp["cramers_v"]
        tmp["effect_size_name"] = "cramers_v"
        parts.append(tmp)
    if not cont.empty:
        tmp = cont.copy()
        tmp["family"] = "continuous"
        tmp["effect_size"] = tmp["epsilon_squared"]
        tmp["effect_size_name"] = "epsilon_squared"
        parts.append(tmp)

    if not parts:
        return cat, cont, pd.DataFrame()

    all_assoc = pd.concat(parts, ignore_index=True, sort=False)
    all_assoc["q_value"] = benjamini_hochberg(all_assoc["p_value"])
    all_assoc["q_value_family"] = np.nan
    all_assoc["q_value_scope"] = np.nan
    all_assoc["q_value_family_scope"] = np.nan

    for family, idx in all_assoc.groupby("family").groups.items():
        all_assoc.loc[idx, "q_value_family"] = benjamini_hochberg(all_assoc.loc[idx, "p_value"])

    for scope, idx in all_assoc.groupby("analysis_scope").groups.items():
        all_assoc.loc[idx, "q_value_scope"] = benjamini_hochberg(all_assoc.loc[idx, "p_value"])

    for (_scope, _family), idx in all_assoc.groupby(["analysis_scope", "family"]).groups.items():
        all_assoc.loc[idx, "q_value_family_scope"] = benjamini_hochberg(all_assoc.loc[idx, "p_value"])

    correction_cols = ["q_value", "q_value_family", "q_value_scope", "q_value_family_scope"]
    correction_map = all_assoc.set_index("test_id")[correction_cols]
    if not cat.empty:
        for column in correction_cols:
            cat[column] = cat["test_id"].map(correction_map[column])
    if not cont.empty:
        for column in correction_cols:
            cont[column] = cont["test_id"].map(correction_map[column])
    return cat, cont, all_assoc


def plot_effect_heatmap(
    associations: pd.DataFrame,
    *,
    effect_column: str,
    output_path: Path,
    title: str,
    q_column: str = "q_value_scope",
) -> None:
    """Plot primary method x metadata effect sizes."""

    if associations.empty:
        return
    pivot = associations.pivot(index="method", columns="variable", values=effect_column)
    q_pivot = associations.pivot(index="method", columns="variable", values=q_column)
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


def format_terminal_summary(all_assoc: pd.DataFrame, *, alpha: float) -> str:
    primary = all_assoc.loc[all_assoc["analysis_scope"].eq("primary")].copy()
    sensitivity = all_assoc.loc[all_assoc["analysis_scope"].eq("sensitivity")].copy()
    lines = [
        "TCGA STAD metadata association summary",
        "--------------------------------------",
        "Clusters/configurations were fixed BEFORE metadata testing.",
        f"Primary/sensitivity BH FDR threshold: q_scope < {alpha:g}",
        "",
        f"Primary valid tests:      {int(primary['p_value'].notna().sum())}",
        f"Sensitivity valid tests:  {int(sensitivity['p_value'].notna().sum())}",
        f"Sensitivity configurations represented: {sensitivity['configuration_id'].nunique()}",
        "",
    ]

    significant = primary.loc[
        primary["q_value_scope"].notna() & (primary["q_value_scope"] < alpha)
    ].sort_values(["q_value_scope", "p_value"])
    if significant.empty:
        lines.append("No primary metadata associations passed the primary-scope FDR threshold.")
        return "\n".join(lines)

    lines.append("Primary associations passing FDR:")
    lines.append(
        f"{'Method':<8} {'Variable':<36} {'Test':<28} {'p':>10} {'q':>10} {'Effect':>10}"
    )
    for row in significant.itertuples(index=False):
        lines.append(
            f"{str(row.method):<8} {str(row.variable)[:36]:<36} {str(row.test)[:28]:<28} "
            f"{row.p_value:>10.3g} {row.q_value_scope:>10.3g} {row.effect_size:>10.3f}"
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
    include_sensitivity: bool,
) -> dict[str, object]:
    clustering_dir = Path(clustering_dir)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, best_k = load_inputs(clustering_dir)
    if not include_sensitivity:
        data = data.loc[data["is_primary"]].copy()
        best_k = best_k.loc[best_k["is_primary"]].copy()

    # Configuration manifest is useful directly in Streamlit selectors.
    manifest_columns = [c for c in CONFIG_COLUMNS if c in best_k.columns]
    manifest_extra = [
        c for c in ("silhouette", "calinski_harabasz", "davies_bouldin", "inertia")
        if c in best_k.columns
    ]
    best_k[manifest_columns + manifest_extra].to_csv(
        output_dir / "configuration_manifest.tsv", sep="\t", index=False
    )

    # Metadata availability should count each biological sample once, not once per config.
    sample_level = data.drop_duplicates("sample_id").copy()
    requested_variables = list(dict.fromkeys([*categorical_variables, *continuous_variables]))
    availability = metadata_availability_table(sample_level, requested_variables)
    availability.to_csv(output_dir / "metadata_availability.tsv", sep="\t", index=False)

    missing_requested = availability.loc[~availability["present_in_file"], "variable"].tolist()
    if missing_requested:
        print(
            "Warning: requested metadata columns not found and will be skipped: "
            + ", ".join(missing_requested)
        )

    categorical_rows: list[dict[str, object]] = []
    enrichment_parts: list[pd.DataFrame] = []
    continuous_rows: list[dict[str, object]] = []
    continuous_summary_parts: list[pd.DataFrame] = []

    for config in best_k.itertuples(index=False):
        config_id = str(config.configuration_id)
        subset = data.loc[data["configuration_id"].eq(config_id)].copy()
        if subset.empty:
            raise ValueError(f"No selected assignments found for configuration {config_id}")
        config_meta = _config_metadata(config)

        for variable in categorical_variables:
            if variable not in subset.columns:
                continue
            test_id = f"categorical::{config_id}::{variable}"
            result, enrichment = categorical_association(
                subset["cluster"],
                subset[variable],
                n_permutations=n_permutations,
                rng=_rng_for_test(random_state, test_id),
            )
            categorical_rows.append(
                {"test_id": test_id, **config_meta, "variable": variable, **result}
            )
            if not enrichment.empty:
                for column, value in reversed(list(config_meta.items())):
                    enrichment.insert(0, column, value)
                enrichment.insert(len(config_meta), "variable", variable)
                enrichment_parts.append(enrichment)

        for variable in continuous_variables:
            if variable not in subset.columns:
                continue
            test_id = f"continuous::{config_id}::{variable}"
            result, group_summary = continuous_association(
                subset["cluster"], subset[variable], min_group_size=min_group_size
            )
            continuous_rows.append(
                {"test_id": test_id, **config_meta, "variable": variable, **result}
            )
            if not group_summary.empty:
                for column, value in reversed(list(config_meta.items())):
                    group_summary.insert(0, column, value)
                group_summary.insert(len(config_meta), "variable", variable)
                continuous_summary_parts.append(group_summary)

    categorical = pd.DataFrame(categorical_rows)
    continuous = pd.DataFrame(continuous_rows)
    for frame, columns in (
        (categorical, ["status", "p_value", "cramers_v", "test"]),
        (continuous, ["status", "p_value", "epsilon_squared", "test"]),
    ):
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan

    categorical, continuous, all_assoc = apply_fdr(categorical, continuous)
    categorical.to_csv(output_dir / "categorical_associations.tsv", sep="\t", index=False)
    continuous.to_csv(output_dir / "continuous_associations.tsv", sep="\t", index=False)
    all_assoc.to_csv(output_dir / "all_associations.tsv", sep="\t", index=False)

    primary_assoc = all_assoc.loc[all_assoc["analysis_scope"].eq("primary")].copy()
    sensitivity_assoc = all_assoc.loc[all_assoc["analysis_scope"].eq("sensitivity")].copy()
    primary_assoc.to_csv(output_dir / "primary_all_associations.tsv", sep="\t", index=False)
    sensitivity_assoc.to_csv(output_dir / "sensitivity_all_associations.tsv", sep="\t", index=False)

    enrichment_long = pd.concat(enrichment_parts, ignore_index=True) if enrichment_parts else pd.DataFrame()
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
        output_dir / "continuous_group_summary.tsv", sep="\t", index=False
    )

    # Primary-only static heatmaps preserve a readable canonical report; Streamlit
    # can render any sensitivity configuration from the long tables later.
    valid_cat_primary = (
        categorical.loc[
            categorical["is_primary"].map(_truthy) & categorical["status"].eq("ok")
        ].copy()
        if not categorical.empty
        else pd.DataFrame()
    )
    valid_cont_primary = (
        continuous.loc[
            continuous["is_primary"].map(_truthy) & continuous["status"].eq("ok")
        ].copy()
        if not continuous.empty
        else pd.DataFrame()
    )
    plot_effect_heatmap(
        valid_cat_primary,
        effect_column="cramers_v",
        output_path=figure_dir / "categorical_cramers_v_heatmap.png",
        title="Primary cluster association with categorical metadata",
    )
    plot_effect_heatmap(
        valid_cont_primary,
        effect_column="epsilon_squared",
        output_path=figure_dir / "continuous_epsilon_squared_heatmap.png",
        title="Primary cluster association with continuous metadata",
    )

    summary: dict[str, object] = {
        "input": {
            "samples": int(data["sample_id"].nunique()),
            "clustering_dir": str(clustering_dir),
            "configurations": int(best_k["configuration_id"].nunique()),
            "primary_configurations": int(best_k["is_primary"].map(_truthy).sum()),
            "sensitivity_configurations": int((~best_k["is_primary"].map(_truthy)).sum()),
        },
        "categorical": {
            "requested_variables": list(categorical_variables),
            "valid_tests": int(categorical["p_value"].notna().sum()),
            "n_permutations_for_sparse_tables": int(n_permutations),
        },
        "continuous": {
            "requested_variables": list(continuous_variables),
            "valid_tests": int(continuous["p_value"].notna().sum()),
            "min_group_size": int(min_group_size),
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg",
            "fdr_alpha": float(fdr_alpha),
            "q_value": "all configurations and families",
            "q_value_family": "within categorical/continuous family across all configurations",
            "q_value_scope": "separately within primary vs sensitivity scope",
            "q_value_family_scope": "within family and primary/sensitivity scope",
            "primary_significant_q_scope": int(
                (primary_assoc["q_value_scope"] < fdr_alpha).sum()
            ) if not primary_assoc.empty else 0,
            "sensitivity_significant_q_scope": int(
                (sensitivity_assoc["q_value_scope"] < fdr_alpha).sum()
            ) if not sensitivity_assoc.empty else 0,
        },
        "notes": {
            "cluster_selection_used_metadata": False,
            "parameter_sweep_embeddings_analyzed": bool(include_sensitivity),
            "metadata_used_to_select_embedding_configuration": False,
            "static_heatmaps_show_primary_only": True,
            "survival_time_columns_defaulted_to_continuous": False,
        },
        "outputs": {
            "configuration_manifest": "configuration_manifest.tsv",
            "all_associations": "all_associations.tsv",
            "primary_associations": "primary_all_associations.tsv",
            "sensitivity_associations": "sensitivity_all_associations.tsv",
        },
    }

    with (output_dir / "metadata_association_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(format_terminal_summary(all_assoc, alpha=fdr_alpha))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test primary and cached sensitivity PCA/MDS/t-SNE/UMAP K-means "
            "clusters for association with technical and clinical metadata."
        )
    )
    parser.add_argument(
        "--clustering-dir",
        default="analysis/clustering",
        help="Directory containing updated clustering.py outputs (default: analysis/clustering)",
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
        help="Categorical metadata variables to test. Pass with no values to disable.",
    )
    parser.add_argument(
        "--continuous",
        nargs="*",
        default=list(DEFAULT_CONTINUOUS),
        help="Continuous metadata variables to test. Pass with no values to disable.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Permutations for sparse categorical tables larger than 2x2 (default: 5000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used to derive deterministic per-test permutation seeds (default: 42)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum observations for a cluster to enter a continuous test (default: 2)",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR threshold (default: 0.05)",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Analyze only canonical primary configurations and ignore cached sensitivity clusters.",
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
        include_sensitivity=not args.primary_only,
    )


if __name__ == "__main__":
    main()