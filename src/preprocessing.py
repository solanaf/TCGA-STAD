"""Preprocessing pipeline for TCGA STAD bulk gene-expression data.

The canonical dimensionality-reduction pipeline is:

1. Keep primary tumor samples (TCGA sample type code ``01``).
2. Remove unannotated gene columns whose label begins with ``?|``.
3. Remove genes with any missing expression value among primary tumors.
4. Floor negative normalized-expression values at zero.
5. Remove genes with total expression < 200 across primary tumors.
6. Apply log2(x + 1).
7. Keep genes with sample variance > 0.7 on the log2 scale.
8. Standardize each retained gene across samples (mean 0, variance 1).

Two expression matrices are returned:

``X_expression``
    Log2-transformed expression after QC and low-expression filtering, but
    before variance filtering or z-scoring. Preserve this matrix for later
    biological analyses such as differential expression.

``X_dimred``
    Variance-filtered and gene-wise standardized matrix used as the shared
    input for PCA, MDS, t-SNE, and UMAP.

The script also builds ``sample_info`` by explicitly merging TCGA expression
samples with patient-level metadata, and ``gene_info`` documenting which genes
passed each preprocessing step.

Run directly with:

python preprocessing.py \
    --expression TCGA.STAD.expression.txt \
    --metadata TCGA.STAD.metadata.txt

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = ("patient_id", "sample_id")
DEFAULT_METADATA_MISSING_TOKENS = (
    "[Not Available]",
    "[Not Evaluated]",
    "[Unknown]",
    "[Discrepancy]",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the canonical STAD preprocessing pipeline."""

    primary_sample_type: str = "01"
    min_total_expression: float = 200.0
    min_log_variance: float = 0.7
    variance_ddof: int = 1
    floor_negative_expression: bool = True
    metadata_missing_tokens: tuple[str, ...] = DEFAULT_METADATA_MISSING_TOKENS


@dataclass
class PreprocessingResult:
    """Outputs produced by :func:`preprocess`."""

    X_expression: pd.DataFrame
    X_dimred: pd.DataFrame
    sample_info: pd.DataFrame
    gene_info: pd.DataFrame
    audit: dict[str, Any]


def load_data(
    expression_path: str | Path,
    metadata_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tab-delimited TCGA expression and metadata files."""

    expression = pd.read_csv(expression_path, sep="\t", low_memory=False)
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)

    missing_expression_ids = [c for c in ID_COLUMNS if c not in expression.columns]
    if missing_expression_ids:
        raise ValueError(
            "Expression file is missing required columns: "
            + ", ".join(missing_expression_ids)
        )

    if "patient_id" not in metadata.columns:
        raise ValueError("Metadata file is missing required column: patient_id")

    if expression["sample_id"].duplicated().any():
        duplicates = expression.loc[
            expression["sample_id"].duplicated(keep=False), "sample_id"
        ].unique()
        raise ValueError(
            f"Expression file contains duplicate sample_id values; examples: "
            f"{duplicates[:5].tolist()}"
        )

    if metadata["patient_id"].duplicated().any():
        duplicates = metadata.loc[
            metadata["patient_id"].duplicated(keep=False), "patient_id"
        ].unique()
        raise ValueError(
            f"Metadata contains duplicate patient_id values; examples: "
            f"{duplicates[:5].tolist()}"
        )

    return expression, metadata


def _sample_barcode_part(sample_id: pd.Series, part: int) -> pd.Series:
    """Return one hyphen-delimited part of a TCGA sample barcode."""

    split = sample_id.astype("string").str.split("-")
    bad = split.str.len().lt(part + 1)
    if bad.any():
        examples = sample_id.loc[bad].head().tolist()
        raise ValueError(f"Malformed TCGA sample barcode(s): {examples}")
    return split.str[part]


def add_tcga_barcode_metadata(samples: pd.DataFrame) -> pd.DataFrame:
    """Add technical/sample fields parsed from TCGA aliquot barcodes."""

    out = samples.copy()
    sample_portion = _sample_barcode_part(out["sample_id"], 3)

    # TCGA sample type is the first two characters of the fourth barcode field:
    # e.g. TCGA-XX-YYYY-01A-... -> 01 = primary solid tumor.
    out["sample_type_code"] = sample_portion.str[:2]

    # Tissue Source Site (TSS) is the second field of the TCGA barcode.
    out["tss_code"] = _sample_barcode_part(out["sample_id"], 1)

    # The final aliquot barcode field is the center code. Keep it as a string;
    # it is a technical/batch variable, not a numeric measurement.
    out["center_code"] = out["sample_id"].astype("string").str.rsplit("-", n=1).str[-1]

    return out


def clean_metadata(
    metadata: pd.DataFrame,
    missing_tokens: tuple[str, ...] = DEFAULT_METADATA_MISSING_TOKENS,
) -> pd.DataFrame:
    """Convert TCGA placeholder strings to missing values.

    Columns that are completely empty after cleaning (for example ``Redaction``
    in the supplied metadata file) are removed.
    """

    cleaned = metadata.copy()
    cleaned = cleaned.replace(list(missing_tokens), pd.NA)
    cleaned = cleaned.dropna(axis=1, how="all")
    return cleaned


def _parse_gene_label(label: str) -> tuple[str | None, str | None]:
    """Split a TCGA gene column such as ``TP53|7157`` into symbol and ID."""

    if "|" not in label:
        return label or None, None
    symbol, entrez = label.rsplit("|", 1)
    return (symbol or None), (entrez or None)


def _build_gene_info(raw_gene_columns: list[str]) -> pd.DataFrame:
    parsed = [_parse_gene_label(c) for c in raw_gene_columns]
    info = pd.DataFrame(
        {
            "gene_column": raw_gene_columns,
            "gene_symbol": [x[0] for x in parsed],
            "entrez_id": [x[1] for x in parsed],
        }
    )
    info["annotated"] = ~info["gene_column"].str.startswith("?|", na=False)
    info["complete_in_primary_tumors"] = False
    info["total_expression"] = np.nan
    info["passes_total_expression"] = False
    info["log2_variance"] = np.nan
    info["retained_for_expression"] = False
    info["retained_for_dimred"] = False
    return info


def _merge_sample_metadata(
    tumor_samples: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    sample_info = add_tcga_barcode_metadata(tumor_samples[list(ID_COLUMNS)])
    metadata_clean = clean_metadata(metadata, config.metadata_missing_tokens)

    sample_info = sample_info.merge(
        metadata_clean,
        on="patient_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )

    unmatched = sample_info.loc[sample_info["_merge"] != "both", "patient_id"]
    if not unmatched.empty:
        raise ValueError(
            "Some primary-tumor patients do not have metadata matches: "
            f"{unmatched.head().tolist()}"
        )

    sample_info = sample_info.drop(columns="_merge")
    return sample_info


def preprocess(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Run the canonical TCGA STAD preprocessing pipeline.

    Parameters
    ----------
    expression
        Raw expression dataframe containing ``patient_id``, ``sample_id``, and
        gene-expression columns.
    metadata
        Patient-level TCGA metadata containing ``patient_id``.
    config
        Optional :class:`PreprocessingConfig`. Defaults reproduce the pipeline
        established for this project.
    """

    if config is None:
        config = PreprocessingConfig()

    if config.variance_ddof < 0:
        raise ValueError("variance_ddof must be non-negative")

    # ------------------------------------------------------------------
    # 1. Restrict the analysis to primary solid tumors.
    # ------------------------------------------------------------------
    sample_annotations = add_tcga_barcode_metadata(expression[list(ID_COLUMNS)])
    primary_mask = sample_annotations["sample_type_code"].eq(config.primary_sample_type)
    tumor = expression.loc[primary_mask].copy()

    if tumor.empty:
        raise ValueError(
            f"No samples with TCGA sample type {config.primary_sample_type!r} were found."
        )

    if tumor["patient_id"].duplicated().any():
        # The pipeline expects one primary expression profile per patient so that
        # patient-level metadata can be merged one-to-one.
        duplicate_patients = tumor.loc[
            tumor["patient_id"].duplicated(keep=False), "patient_id"
        ].unique()
        raise ValueError(
            "More than one retained expression sample exists for a patient; "
            f"examples: {duplicate_patients[:5].tolist()}"
        )

    raw_gene_columns = [c for c in expression.columns if c not in ID_COLUMNS]
    gene_info = _build_gene_info(raw_gene_columns)

    # ------------------------------------------------------------------
    # 2. Remove unannotated columns (e.g. ?|100130426).
    # ------------------------------------------------------------------
    annotated_columns = gene_info.loc[gene_info["annotated"], "gene_column"].tolist()
    X = tumor[annotated_columns].apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------
    # 3. Keep only genes complete across all retained primary tumors.
    #    Structured missingness is dropped rather than imputed so it cannot
    #    create artificial geometry in downstream embeddings.
    # ------------------------------------------------------------------
    complete_mask = ~X.isna().any(axis=0)
    complete_columns = X.columns[complete_mask].tolist()
    X = X.loc[:, complete_columns].copy()

    gene_info.loc[
        gene_info["gene_column"].isin(complete_columns),
        "complete_in_primary_tumors",
    ] = True

    # Record negative-value QC before correcting them.
    negative_mask = X.lt(0)
    negative_value_count = int(negative_mask.to_numpy().sum())
    genes_with_negative_values = int(negative_mask.any(axis=0).sum())
    samples_with_negative_values = int(negative_mask.any(axis=1).sum())

    # ------------------------------------------------------------------
    # 4. Floor small negative normalized-expression values at zero.
    # ------------------------------------------------------------------
    if config.floor_negative_expression:
        X = X.clip(lower=0)
    elif (X.to_numpy() <= -1).any():
        raise ValueError(
            "Expression contains values <= -1, so log2(x + 1) is undefined. "
            "Enable floor_negative_expression or choose another transformation."
        )

    # ------------------------------------------------------------------
    # 5. Remove very low-expression genes using the original project cutoff.
    # ------------------------------------------------------------------
    total_expression = X.sum(axis=0)
    passes_total = total_expression.ge(config.min_total_expression)
    expression_columns = total_expression.index[passes_total].tolist()
    X = X.loc[:, expression_columns].copy()

    gene_info_indexed = gene_info.set_index("gene_column")
    gene_info_indexed.loc[total_expression.index, "total_expression"] = total_expression
    gene_info_indexed.loc[passes_total.index, "passes_total_expression"] = passes_total
    gene_info_indexed.loc[expression_columns, "retained_for_expression"] = True

    # ------------------------------------------------------------------
    # 6. Log transform. This matrix is retained separately for later DEG work.
    # ------------------------------------------------------------------
    X_log = np.log2(X + 1.0)
    X_log.index = tumor["sample_id"].to_numpy()
    X_log.index.name = "sample_id"

    # ------------------------------------------------------------------
    # 7. Keep variable genes. Use sample variance (ddof=1) by default to match
    #    the original pandas-based thresholding used in the class workflow.
    # ------------------------------------------------------------------
    log_variance = X_log.var(axis=0, ddof=config.variance_ddof)
    passes_variance = log_variance.gt(config.min_log_variance)
    dimred_columns = log_variance.index[passes_variance].tolist()

    gene_info_indexed.loc[log_variance.index, "log2_variance"] = log_variance
    gene_info_indexed.loc[dimred_columns, "retained_for_dimred"] = True

    X_for_scaling = X_log.loc[:, dimred_columns]

    # ------------------------------------------------------------------
    # 8. Standardize genes across samples. Rows remain samples; columns remain
    #    genes. StandardScaler therefore centers/scales each gene independently.
    # ------------------------------------------------------------------
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled_array = scaler.fit_transform(X_for_scaling)
    X_dimred = pd.DataFrame(
        X_scaled_array,
        index=X_for_scaling.index.copy(),
        columns=X_for_scaling.columns.copy(),
    )

    gene_info = gene_info_indexed.reset_index()

    # Merge metadata explicitly after sample filtering; this is one-to-one for
    # the retained primary tumors in the supplied STAD files.
    sample_info = _merge_sample_metadata(
        tumor_samples=tumor[list(ID_COLUMNS)],
        metadata=metadata,
        config=config,
    )

    # Match sample_info row order to both expression matrices.
    sample_info = sample_info.set_index("sample_id").loc[X_log.index].reset_index()

    n_complete = len(complete_columns)
    n_expression = len(expression_columns)
    n_dimred = len(dimred_columns)
    n_values_complete = X.shape[0] * n_complete

    audit: dict[str, Any] = {
        "config": asdict(config),
        "raw": {
            "samples": int(expression.shape[0]),
            "gene_features": int(len(raw_gene_columns)),
            "metadata_patients": int(metadata["patient_id"].nunique()),
        },
        "sample_filtering": {
            "primary_sample_type": config.primary_sample_type,
            "primary_tumor_samples": int(tumor.shape[0]),
            "primary_tumor_patients": int(tumor["patient_id"].nunique()),
            "metadata_matches": int(sample_info.shape[0]),
        },
        "gene_filtering": {
            "unannotated_removed": int(len(raw_gene_columns) - len(annotated_columns)),
            "annotated_genes": int(len(annotated_columns)),
            "genes_with_missing_values_removed": int(len(annotated_columns) - n_complete),
            "complete_genes": int(n_complete),
            "low_expression_removed": int(n_complete - n_expression),
            "expression_matrix_genes": int(n_expression),
            "low_variance_removed": int(n_expression - n_dimred),
            "dimred_matrix_genes": int(n_dimred),
        },
        "negative_expression_qc": {
            "negative_values_before_flooring": negative_value_count,
            "genes_with_negative_values": genes_with_negative_values,
            "samples_with_negative_values": samples_with_negative_values,
            "percent_negative_values_among_complete_genes": (
                100.0 * negative_value_count / n_values_complete
                if n_values_complete
                else 0.0
            ),
        },
        "outputs": {
            "X_expression_shape": [int(v) for v in X_log.shape],
            "X_dimred_shape": [int(v) for v in X_dimred.shape],
            "sample_info_shape": [int(v) for v in sample_info.shape],
        },
    }

    return PreprocessingResult(
        X_expression=X_log,
        X_dimred=X_dimred,
        sample_info=sample_info,
        gene_info=gene_info,
        audit=audit,
    )


def format_audit(audit: dict[str, Any]) -> str:
    """Return a concise human-readable preprocessing summary."""

    raw = audit["raw"]
    samples = audit["sample_filtering"]
    genes = audit["gene_filtering"]
    neg = audit["negative_expression_qc"]
    outputs = audit["outputs"]

    return "\n".join(
        [
            "TCGA STAD preprocessing summary",
            "--------------------------------",
            f"Raw samples:                         {raw['samples']:,}",
            f"Raw gene features:                   {raw['gene_features']:,}",
            f"Primary tumor samples retained:       {samples['primary_tumor_samples']:,}",
            f"Unannotated genes removed:             {genes['unannotated_removed']:,}",
            f"Genes with missing values removed:     {genes['genes_with_missing_values_removed']:,}",
            f"Complete annotated genes:             {genes['complete_genes']:,}",
            f"Negative values floored to zero:      {neg['negative_values_before_flooring']:,}",
            f"Low-expression genes removed:           {genes['low_expression_removed']:,}",
            f"Genes in X_expression:                {outputs['X_expression_shape'][1]:,}",
            f"Low-variance genes removed:           {genes['low_variance_removed']:,}",
            f"Genes in X_dimred:                     {outputs['X_dimred_shape'][1]:,}",
            f"Final samples:                           {outputs['X_dimred_shape'][0]:,}",
        ]
    )


def save_results(result: PreprocessingResult, output_dir: str | Path) -> None:
    """Save preprocessing outputs using dependency-light compressed TSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.X_expression.to_csv(
        output_dir / "X_expression.tsv.gz",
        sep="\t",
        compression="gzip",
    )
    result.X_dimred.to_csv(
        output_dir / "X_dimred.tsv.gz",
        sep="\t",
        compression="gzip",
    )
    result.sample_info.to_csv(output_dir / "sample_info.tsv", sep="\t", index=False)
    result.gene_info.to_csv(output_dir / "gene_info.tsv", sep="\t", index=False)

    with (output_dir / "preprocessing_audit.json").open("w", encoding="utf-8") as f:
        json.dump(result.audit, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess TCGA STAD expression data for dimensionality reduction."
    )
    parser.add_argument(
        "--expression",
        default="TCGA.STAD.expression.txt",
        help="Path to TCGA.STAD.expression.txt",
    )
    parser.add_argument(
        "--metadata",
        default="TCGA.STAD.metadata.txt",
        help="Path to TCGA.STAD.metadata.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="processed",
        help="Directory for processed matrices and audit files (default: processed)",
    )
    parser.add_argument(
        "--min-total-expression",
        type=float,
        default=200.0,
        help="Minimum total expression across primary tumors (default: 200)",
    )
    parser.add_argument(
        "--min-log-variance",
        type=float,
        default=0.7,
        help="Minimum sample variance after log2(x+1) (default: 0.7)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run preprocessing and print the audit without writing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = PreprocessingConfig(
        min_total_expression=args.min_total_expression,
        min_log_variance=args.min_log_variance,
    )

    expression, metadata = load_data(args.expression, args.metadata)
    result = preprocess(expression, metadata, config)

    print(format_audit(result.audit))

    if not args.no_save:
        save_results(result, args.output_dir)
        print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()