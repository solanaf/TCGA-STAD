from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from utils import charts
from utils.data import (
    configuration_manifest,
    coord_columns,
    embedding_coordinates,
    maybe_json,
    maybe_tsv,
    merge_metadata,
    safe_numeric,
    sample_info,
    truthy,
)


STEPS = (
    "Overview",
    "Preprocessing",
    "Dimensionality reduction",
    "Clustering",
    "Metadata association",
)

METHOD_NAMES = {"pca": "PCA", "mds": "MDS", "tsne": "t-SNE", "umap": "UMAP"}


def _sidebar() -> tuple[str, Path]:
    """Render Project 1 workflow navigation and use the launch directory as root."""

    st.sidebar.caption("Project 1")
    step = st.sidebar.radio(
        "Workflow step",
        STEPS,
        key="project1_step",
        help=(
            "Move through the analysis in the same order as the computational pipeline. "
            "The app reads cached outputs so changing a cached configuration is immediate."
        ),
    )

    # Streamlit is launched from the TCGA-STAD repository root, so the project
    # root is always the current working directory. This is intentionally not
    # exposed as a UI control.
    root = Path.cwd().resolve()
    return step, root


def _missing(root: Path, paths: list[str]) -> bool:
    missing = [p for p in paths if not (root / p).exists()]
    if not missing:
        return False
    st.warning("Missing cached file(s):\n\n" + "\n".join(f"- `{p}`" for p in missing))
    return True


def _config_badge(row: pd.Series | None) -> None:
    if row is None:
        return
    primary = bool(truthy(pd.Series([row.get("is_primary", False)])).iloc[0])
    if primary:
        st.caption("**Primary configuration** · canonical analysis")
    else:
        st.caption("**Sensitivity configuration** · secondary/exploratory analysis")


def _metadata_color_control(metadata: pd.DataFrame | None, key: str) -> str | None:
    if metadata is None:
        return None
    excluded = {"patient_id", "sample_id"}
    options = ["None"] + [c for c in metadata.columns if c not in excluded]
    chosen = st.selectbox(
        "Color points by",
        options,
        key=key,
        help=(
            "Overlay a technical or clinical metadata variable without changing the embedding. "
            "This is visualization only; it does not tune dimensionality reduction."
        ),
    )
    return None if chosen == "None" else chosen


def render() -> None:
    step, root = _sidebar()

    st.title("TCGA-STAD · Subgroups of Gene Expression Data")
    st.caption(
        "Project 1 · preprocessing → dimensionality reduction → clustering → metadata association"
    )
    st.divider()

    if step == "Overview":
        render_overview(root)
    elif step == "Preprocessing":
        render_preprocessing(root)
    elif step == "Dimensionality reduction":
        render_dimensionality_reduction(root)
    elif step == "Clustering":
        render_clustering(root)
    else:
        render_metadata(root)


def render_overview(root: Path) -> None:
    st.subheader(
        "Analysis workflow",
        help="Each stage is kept separate so later biological interpretation cannot leak backward into parameter selection.",
    )
    st.write(
        "This page is a cache-backed gene expression explorer of the TCGA-STAD dataset. "
        "Use the workflow selector in the sidebar to inspect what happened at each stage and "
        "switch among the parameter combinations that were already computed."
    )

    files = {
        "Preprocessing": "processed/preprocessing_audit.json",
        "PCA": "analysis/pca/pca_scores.tsv.gz",
        "MDS": "analysis/mds/mds_scores.tsv.gz",
        "t-SNE sweep": "analysis/tsne/tsne_sensitivity_scores.tsv.gz",
        "UMAP sweep": "analysis/umap/umap_sensitivity_scores.tsv.gz",
        "Clustering": "analysis/clustering/kmeans_diagnostics.tsv",
        "Metadata association": "analysis/metadata_association/all_associations.tsv",
    }
    cols = st.columns(4)
    for i, (label, relative) in enumerate(files.items()):
        exists = (root / relative).exists()
        cols[i % 4].metric(label, "Cached" if exists else "Missing")

    st.markdown("#### Design principle")
    st.info(
        "The primary Euclidean/default analyses stay explicitly marked as primary. "
        "Alternative t-SNE/UMAP configurations are available as sensitivity analyses, "
        "but metadata are never used to choose the embedding configuration or the number of clusters.",
        icon="ℹ️",
    )


def render_preprocessing(root: Path) -> None:
    st.subheader(
        "Preprocessing",
        help=(
            "The canonical pipeline retains primary tumors, removes problematic genes, floors negative "
            "normalized-expression values at zero, applies log2(x+1), filters variable genes, and z-scores genes."
        ),
    )
    if _missing(root, ["processed/preprocessing_audit.json", "processed/gene_info.tsv"]):
        return
    audit = maybe_json(root, "processed/preprocessing_audit.json") or {}
    gene_info = maybe_tsv(root, "processed/gene_info.tsv")

    raw = audit.get("raw", {})
    samples = audit.get("sample_filtering", {})
    genes = audit.get("gene_filtering", {})
    neg = audit.get("negative_expression_qc", {})
    outputs = audit.get("outputs", {})
    config = audit.get("config", {})

    cols = st.columns(5)
    cols[0].metric("Raw samples", f"{raw.get('samples', 0):,}")
    cols[1].metric("Primary tumors", f"{samples.get('primary_tumor_samples', 0):,}")
    cols[2].metric("Raw genes", f"{raw.get('gene_features', 0):,}")
    cols[3].metric("Expression genes", f"{genes.get('expression_matrix_genes', 0):,}")
    cols[4].metric("Dim-red genes", f"{genes.get('dimred_matrix_genes', 0):,}")

    st.markdown("#### Canonical preprocessing settings")

    total_expression = float(config.get("min_total_expression", 200.0))
    min_variance = float(config.get("min_log_variance", 0.7))
    floor_negative = bool(config.get("floor_negative_expression", True))

    preprocessing_facts = [
        (
            "Minimum total expression",
            f"{total_expression:g} summed RSEM-normalized expression units",
            "Genes with total expression below this threshold across retained primary tumors were removed. "
            "This suppresses very low-signal genes before log transformation.",
        ),
        (
            "Negative expression handling",
            "Floored at zero" if floor_negative else "Not floored",
            "Small negative values in the normalized-expression source were set to zero before log2(x+1). "
            "This prevents negative inputs from creating problems for the log transform.",
        ),
        (
            "Expression transformation",
            "log2(x + 1)",
            "The log transformation compresses the large expression range and reduces right-skew while "
            "remaining defined for zero-valued expression.",
        ),
        (
            "Minimum log2 variance",
            f"{min_variance:g}",
            "After log2(x+1), genes had to exceed this sample-variance threshold to enter X_dimred. "
            "This focuses the embedding on genes that vary across tumors.",
        ),
        (
            "Final scaling",
            "Gene-wise z-score standardization",
            "Each retained gene was centered and scaled across samples so genes with larger numerical ranges "
            "would not dominate the downstream distance geometry solely because of scale.",
        ),
    ]

    for label, value, explanation in preprocessing_facts:
        label_col, value_col, info_col = st.columns([2.4, 2.2, 0.35], vertical_alignment="center")
        label_col.markdown(f"**{label}**")
        value_col.write(value)
        with info_col.popover("ⓘ"):
            st.write(explanation)

    st.markdown("#### Filtering audit")
    audit_table = pd.DataFrame(
        [
            ("Unannotated genes removed", genes.get("unannotated_removed")),
            ("Genes with missing values removed", genes.get("genes_with_missing_values_removed")),
            ("Negative values floored", neg.get("negative_values_before_flooring")),
            ("Low-expression genes removed", genes.get("low_expression_removed")),
            ("Low-variance genes removed", genes.get("low_variance_removed")),
        ],
        columns=["Step", "Count"],
    )
    st.dataframe(audit_table, hide_index=True, use_container_width=True)

    with st.popover("ⓘ Why these preprocessing steps?"):
        st.markdown(
            """
            **Primary tumors only:** keeps the biological comparison focused on one specimen type.  
            **Remove unannotated genes:** avoids features without usable gene identity.  
            **Drop genes with missing values:** avoids imputation creating artificial distances between samples.  
            **Floor negatives at zero:** makes the normalized-expression values compatible with `log2(x+1)`.  
            **Low-expression filter:** removes features dominated by near-zero signal.  
            **Log transform:** compresses large expression ranges and reduces right skew.  
            **Variance filter:** focuses dimensionality reduction on genes that vary across samples.  
            **Gene-wise z-scoring:** prevents high-scale genes from dominating Euclidean geometry simply because of scale.
            """
        )

    if gene_info is not None:
        st.markdown("#### Gene-level preprocessing table")
        display_cols = [c for c in [
            "gene_symbol", "entrez_id", "annotated", "complete_in_primary_tumors",
            "total_expression", "passes_total_expression", "log2_variance",
            "retained_for_expression", "retained_for_dimred",
        ] if c in gene_info.columns]
        st.dataframe(gene_info[display_cols], use_container_width=True, height=390)


def render_dimensionality_reduction(root: Path) -> None:
    st.subheader(
        "Dimensionality reduction",
        help=(
            "Switch among PCA, MDS, t-SNE, and UMAP. t-SNE and UMAP controls only show configurations that "
            "were actually cached, so changing a selector does not trigger expensive recomputation."
        ),
    )
    metadata = sample_info(root)
    method_options = list(METHOD_NAMES)
    method_key = st.selectbox(
        "Method",
        method_options,
        index=method_options.index("pca") if "pca" in method_options else 0,
        key="p1_dimred_method",
        format_func=lambda x: METHOD_NAMES[x],
        help="All four methods consume the same canonical X_dimred matrix in the primary comparison.",
    )

    if method_key == "pca":
        render_pca(root, metadata)
    elif method_key == "mds":
        render_mds(root, metadata)
    elif method_key == "tsne":
        render_tsne(root, metadata)
    else:
        render_umap(root, metadata)


def render_pca(root: Path, metadata: pd.DataFrame | None) -> None:
    coords = embedding_coordinates(root, "pca")
    variance = maybe_tsv(root, "analysis/pca/pca_variance.tsv")
    if coords is None:
        st.warning("PCA cache not found.")
        return
    pc_cols = [c for c in coords.columns if c.startswith("PC")]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X component", pc_cols, index=0, help="Principal components are orthogonal axes ordered by explained variance.")
    y = c2.selectbox("Y component", pc_cols, index=min(1, len(pc_cols)-1))
    color = _metadata_color_control(metadata, "pca_color")
    data = merge_metadata(coords, metadata)
    st.altair_chart(charts.scatter(data, x, y, color=color, title=f"PCA: {x} vs {y}"), use_container_width=True)

    if variance is not None:
        selected = variance.loc[variance["component"].isin([x, y])]
        st.dataframe(selected, hide_index=True, use_container_width=True)
        if {"component", "explained_variance_percent"}.issubset(variance.columns):
            n = min(25, len(variance))
            st.altair_chart(
                charts.line(variance.head(n), "component", "explained_variance_percent", title="PCA scree plot"),
                use_container_width=True,
            )
    with st.popover("ⓘ PCA interpretation"):
        st.write(
            "PCA is linear and global: each component is a weighted combination of genes, and the variance "
            "table quantifies how much standardized expression variation each axis explains."
        )


def render_mds(root: Path, metadata: pd.DataFrame | None) -> None:
    coords = embedding_coordinates(root, "mds")
    diag = maybe_tsv(root, "analysis/mds/mds_dimension_diagnostics.tsv")
    summary = maybe_json(root, "analysis/mds/mds_summary.json") or {}
    if coords is None:
        st.warning("MDS cache not found.")
        return
    data = merge_metadata(coords, metadata)
    x, y = coord_columns("mds", data)
    color = _metadata_color_control(metadata, "mds_color")
    st.altair_chart(charts.scatter(data, x, y, color=color, title="MDS embedding"), use_container_width=True)

    info = summary.get("mds", {})
    c = st.columns(4)
    c[0].metric("Distance metric", str(info.get("distance_metric", "—")))
    c[1].metric("Stress-1", safe_numeric(info.get("primary_stress1")))
    c[2].metric("Distance Pearson r", safe_numeric(info.get("primary_pearson_r")))
    c[3].metric("Distance Spearman ρ", safe_numeric(info.get("primary_spearman_rho")))
    if diag is not None:
        st.dataframe(diag, hide_index=True, use_container_width=True)
    with st.popover("ⓘ MDS interpretation"):
        st.write(
            "Metric MDS attempts to preserve pairwise sample dissimilarities in a low-dimensional space. "
            "Stress summarizes distortion: lower stress means the displayed geometry better reproduces the original distances."
        )


def render_tsne(root: Path, metadata: pd.DataFrame | None) -> None:
    coords = embedding_coordinates(root, "tsne")
    diag = maybe_tsv(root, "analysis/tsne/tsne_sensitivity_diagnostics.tsv")
    if coords is None or diag is None:
        st.warning("Updated t-SNE sensitivity cache not found.")
        return
    c1, c2 = st.columns(2)
    metrics = list(dict.fromkeys(diag["metric"].astype(str)))
    metric_default = "euclidean" if "euclidean" in metrics else metrics[0]
    metric = c1.selectbox(
        "Distance metric",
        metrics,
        index=metrics.index(metric_default),
        key="p1_tsne_metric",
        help="Defines similarity in the original high-dimensional gene-expression space before t-SNE constructs local neighborhoods.",
    )
    available = diag.loc[diag["metric"].astype(str).eq(metric)]
    perplexities = sorted(available["perplexity"].astype(float).unique())
    perplexity_default = 30.0 if 30.0 in perplexities else perplexities[0]
    perplexity = c2.selectbox(
        "Perplexity",
        perplexities,
        index=perplexities.index(perplexity_default),
        key="p1_tsne_perplexity",
        help="Controls the effective neighborhood scale. Smaller values emphasize more local structure; larger values consider broader neighborhoods.",
    )
    selected = available.loc[available["perplexity"].astype(float).eq(float(perplexity))].iloc[0]
    config_id = str(selected.get("configuration_id", ""))
    if not config_id:
        config_id = f"tsne__metric-{metric}__perplexity-{f'{float(perplexity):g}'.replace('.', 'p')}"
    plot_df = embedding_coordinates(root, "tsne", config_id)
    if plot_df is None or plot_df.empty:
        # Backward-compatible filtering if IDs differ in a historical cache.
        plot_df = coords.loc[
            coords["metric"].astype(str).eq(metric) & coords["perplexity"].astype(float).eq(float(perplexity))
        ].copy()
    data = merge_metadata(plot_df, metadata)
    color = _metadata_color_control(metadata, "tsne_color")
    st.altair_chart(charts.scatter(data, "tSNE1", "tSNE2", color=color, title=f"t-SNE · {metric} · perplexity={perplexity:g}"), use_container_width=True)
    cols = st.columns(5)
    for col, name, label in zip(
        cols,
        ["kl_divergence", "trustworthiness_k15", "trustworthiness_k30", "mean_jaccard_k15", "mean_jaccard_k30"],
        ["KL divergence", "Trustworthiness k=15", "Trustworthiness k=30", "Jaccard k=15", "Jaccard k=30"],
    ):
        col.metric(label, safe_numeric(selected.get(name)))
    with st.popover("ⓘ t-SNE parameters & diagnostics"):
        st.markdown(
            """
            **Metric** changes which samples count as neighbors in the original space.  
            **Perplexity** changes the neighborhood scale used by t-SNE.  
            **KL divergence** is the t-SNE optimization objective; lower is better *within comparable setups*, but it is not a biological score.  
            **Trustworthiness/Jaccard** measure how well local neighborhoods survive the projection.
            """
        )


def render_umap(root: Path, metadata: pd.DataFrame | None) -> None:
    diag = maybe_tsv(root, "analysis/umap/umap_parameter_diagnostics.tsv")
    coords = embedding_coordinates(root, "umap")
    if diag is None or coords is None:
        st.warning("Updated UMAP sensitivity cache not found.")
        return
    c1, c2, c3 = st.columns(3)
    metrics = list(dict.fromkeys(diag["metric"].astype(str)))
    metric_default = "euclidean" if "euclidean" in metrics else metrics[0]
    metric = c1.selectbox(
        "Distance metric",
        metrics,
        index=metrics.index(metric_default),
        key="p1_umap_metric",
        help="Defines high-dimensional sample distance before the UMAP neighborhood graph is constructed.",
    )
    d1 = diag.loc[diag["metric"].astype(str).eq(metric)]
    neighbors = sorted(d1["n_neighbors"].astype(int).unique())
    nn_default = 15 if 15 in neighbors else neighbors[0]
    nn = c2.selectbox(
        "n_neighbors",
        neighbors,
        index=neighbors.index(nn_default),
        key="p1_umap_n_neighbors",
        help="Controls how local versus global the UMAP neighborhood graph is. Smaller values focus more strongly on local neighborhoods.",
    )
    d2 = d1.loc[d1["n_neighbors"].astype(int).eq(int(nn))]
    min_dists = sorted(d2["min_dist"].astype(float).unique())
    md_default = 0.1 if 0.1 in min_dists else min_dists[0]
    md = c3.selectbox(
        "min_dist",
        min_dists,
        index=min_dists.index(md_default),
        key="p1_umap_min_dist",
        help="Controls how tightly UMAP is allowed to pack nearby points in the displayed embedding.",
    )
    selected = d2.loc[d2["min_dist"].astype(float).eq(float(md))].iloc[0]
    config_id = str(selected.get("configuration_id", ""))
    if not config_id:
        config_id = f"umap__metric-{metric}__nn-{int(nn)}__mindist-{f'{float(md):g}'.replace('.', 'p')}"
    plot_df = embedding_coordinates(root, "umap", config_id)
    if plot_df is None or plot_df.empty:
        plot_df = coords.loc[
            coords["metric"].astype(str).eq(metric)
            & coords["n_neighbors"].astype(int).eq(int(nn))
            & coords["min_dist"].astype(float).eq(float(md))
        ].copy()
    data = merge_metadata(plot_df, metadata)
    color = _metadata_color_control(metadata, "umap_color")
    st.altair_chart(charts.scatter(data, "UMAP1", "UMAP2", color=color, title=f"UMAP · {metric} · n_neighbors={nn} · min_dist={md:g}"), use_container_width=True)
    cols = st.columns(4)
    for col, name, label in zip(
        cols,
        ["trustworthiness_k15", "trustworthiness_k30", "mean_jaccard_k15", "mean_jaccard_k30"],
        ["Trustworthiness k=15", "Trustworthiness k=30", "Jaccard k=15", "Jaccard k=30"],
    ):
        col.metric(label, safe_numeric(selected.get(name)))
    _config_badge(selected)
    with st.popover("ⓘ UMAP parameters & diagnostics"):
        st.markdown(
            """
            **Metric** determines original-space distance.  
            **n_neighbors** controls how local the learned manifold is.  
            **min_dist** controls visual packing in the low-dimensional embedding.  
            The neighborhood diagnostics are metric-aware, so comparisons across metrics should be interpreted as sensitivity analyses rather than a single universal leaderboard.
            """
        )


def _configuration_selector_from_manifest(
    manifest: pd.DataFrame,
    key_prefix: str,
) -> pd.Series | None:
    """Select one cached embedding configuration for the current workflow step."""

    if manifest is None or manifest.empty:
        st.warning("Embedding configuration manifest is missing.")
        return None

    method_options = list(dict.fromkeys(manifest["method_key"].astype(str)))
    default_method = "pca" if "pca" in method_options else method_options[0]

    c1, c2 = st.columns([1, 2])
    method_key = c1.selectbox(
        "Embedding method",
        method_options,
        index=method_options.index(default_method),
        key=f"p1_{key_prefix}_method",
        format_func=lambda x: METHOD_NAMES.get(x, x),
    )

    subset = manifest.loc[manifest["method_key"].astype(str).eq(method_key)].copy()
    if subset.empty:
        st.warning(f"No cached configurations found for {METHOD_NAMES.get(method_key, method_key)}.")
        return None

    labels = {
        str(row.configuration_id): (
            f"{'★ ' if bool(truthy(pd.Series([row.is_primary])).iloc[0]) else ''}{row.configuration_label}"
        )
        for row in subset.itertuples(index=False)
    }
    config_options = list(labels)
    primary_ids = [
        str(row.configuration_id)
        for row in subset.itertuples(index=False)
        if bool(truthy(pd.Series([row.is_primary])).iloc[0])
    ]
    default_config = primary_ids[0] if primary_ids else config_options[0]

    config_id = c2.selectbox(
        "Embedding configuration",
        config_options,
        index=config_options.index(default_config),
        key=f"p1_{key_prefix}_config",
        format_func=lambda x: labels[x],
        help="★ marks the canonical primary configuration. All other entries are cached sensitivity analyses.",
    )

    return subset.loc[subset["configuration_id"].astype(str).eq(config_id)].iloc[0]

def _configuration_selector(root: Path, key_prefix: str) -> tuple[pd.Series, pd.DataFrame] | tuple[None, None]:
    manifest = configuration_manifest(root)
    if manifest is None or manifest.empty:
        st.warning("Embedding configuration manifest is missing. Run the updated clustering.py first.")
        return None, None
    row = _configuration_selector_from_manifest(manifest, key_prefix)
    return (row, manifest) if row is not None else (None, None)


def render_clustering(root: Path) -> None:
    st.subheader(
        "K-means clustering",
        help=(
            "K-means was run independently on every cached 2-D embedding for k=2–10. "
            "The best k for each configuration was selected by maximum silhouette score, without using metadata."
        ),
    )
    if _missing(root, ["analysis/clustering/kmeans_diagnostics.tsv", "analysis/clustering/kmeans_assignments_long.tsv.gz", "analysis/clustering/best_k_summary_all.tsv"]):
        return
    config_row, _ = _configuration_selector(root, "cluster")
    if config_row is None:
        return
    config_id = str(config_row["configuration_id"])
    method_key = str(config_row["method_key"])

    diag = maybe_tsv(root, "analysis/clustering/kmeans_diagnostics.tsv")
    assignments = maybe_tsv(root, "analysis/clustering/kmeans_assignments_long.tsv.gz")
    best = maybe_tsv(root, "analysis/clustering/best_k_summary_all.tsv")
    d = diag.loc[diag["configuration_id"].astype(str).eq(config_id)].sort_values("k")
    best_row = best.loc[best["configuration_id"].astype(str).eq(config_id)].iloc[0]
    best_k = int(best_row["k"])
    ks = d["k"].astype(int).tolist()
    st.caption(f"Best-by-silhouette for this configuration is k={best_k}.")
    k = st.number_input(
        "Number of clusters (k)",
        min_value=min(ks),
        max_value=max(ks),
        value=best_k,
        width=150,
        step=1,
        help=(
            "The cached grid contains every tested k. "
            "The recommended/default k is the one with the maximum "
            "silhouette score for this embedding configuration."
        ),
    )
    selected_diag = d.loc[d["k"].astype(int).eq(int(k))].iloc[0]
    cols = st.columns(5)
    for col, name, label in zip(
        cols,
        ["silhouette", "calinski_harabasz", "davies_bouldin", "min_cluster_size", "max_cluster_size"],
        ["Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Smallest cluster", "Largest cluster"],
    ):
        col.metric(label, safe_numeric(selected_diag.get(name), 3 if "size" not in name else 0))

    st.altair_chart(charts.line(d, "k", "silhouette", title="Silhouette across k", highlight_x=best_k), use_container_width=True)

    labels = assignments.loc[
        assignments["configuration_id"].astype(str).eq(config_id) & assignments["k"].astype(int).eq(int(k)),
        ["sample_id", "cluster"],
    ].copy()
    coords = embedding_coordinates(root, method_key, config_id)
    if coords is not None and not coords.empty:
        x, y = coord_columns(method_key, coords)
        plot_df = coords.merge(labels, on="sample_id", how="inner", validate="one_to_one")
        plot_df["cluster"] = plot_df["cluster"].astype(str)
        st.altair_chart(charts.scatter(plot_df, x, y, color="cluster", title=f"{METHOD_NAMES.get(method_key, method_key)} · K-means k={k}", tooltip_extra=["cluster"]), use_container_width=True)

    with st.popover("ⓘ Why silhouette selects k"):
        st.write(
            "Silhouette compares within-cluster cohesion with separation from neighboring clusters. "
            "We use it to select k before looking at clinical or technical metadata, which reduces the risk of choosing a cluster solution simply because it matches a desired metadata variable."
        )


def render_metadata(root: Path) -> None:
    st.subheader(
        "Metadata association",
        help=(
            "Tests whether the already-fixed best-k clusters are associated with technical or clinical metadata. "
            "Metadata are not used to choose the embedding configuration or k."
        ),
    )
    if _missing(root, ["analysis/metadata_association/all_associations.tsv", "analysis/metadata_association/configuration_manifest.tsv"]):
        return
    assoc = maybe_tsv(root, "analysis/metadata_association/all_associations.tsv")
    manifest = maybe_tsv(root, "analysis/metadata_association/configuration_manifest.tsv")
    enrichment = maybe_tsv(root, "analysis/metadata_association/categorical_enrichment_long.tsv.gz")
    continuous_summary = maybe_tsv(root, "analysis/metadata_association/continuous_group_summary.tsv")
    assignment_meta = maybe_tsv(root, "analysis/clustering/best_cluster_assignments_all_with_metadata.tsv.gz")

    if assoc is None or assoc.empty:
        st.warning("No metadata association results found.")
        return
    if manifest is None or manifest.empty:
        manifest = assoc.drop_duplicates("configuration_id")

    config_row = _configuration_selector_from_manifest(manifest, "meta")
    if config_row is None:
        return
    method_key = str(config_row["method_key"])
    config_id = str(config_row["configuration_id"])
    _config_badge(config_row)

    subset = assoc.loc[assoc["configuration_id"].astype(str).eq(config_id)].copy()
    families = [f for f in ["categorical", "continuous"] if f in set(subset["family"].astype(str))]
    family = st.radio(
        "Metadata type", families, horizontal=True,
        help="Categorical variables use Fisher/chi-square-style independence tests; continuous variables use Kruskal–Wallis.",
    )
    family_df = subset.loc[subset["family"].astype(str).eq(family)]
    variable = st.selectbox("Metadata variable", sorted(family_df["variable"].astype(str).unique()))
    row = family_df.loc[family_df["variable"].astype(str).eq(variable)].iloc[0]

    c = st.columns(6)
    c[0].metric("Best k", int(row.get("k")) if pd.notna(row.get("k")) else "—")
    c[1].metric("Test", str(row.get("test", "—")))
    c[2].metric("p-value", safe_numeric(row.get("p_value"), 4))
    c[3].metric("q (scope)", safe_numeric(row.get("q_value_scope"), 4))
    effect_name = "Cramér's V" if family == "categorical" else "ε²"
    c[4].metric(effect_name, safe_numeric(row.get("effect_size"), 3))
    c[5].metric("N", int(row.get("n")) if pd.notna(row.get("n")) else "—")

    q = row.get("q_value_scope")
    if pd.notna(q):
        if float(q) < 0.05:
            st.success("This association passes the within-scope BH FDR threshold (q < 0.05).")
        else:
            st.info("This association does not pass the within-scope BH FDR threshold (q < 0.05).")

    if family == "categorical" and enrichment is not None and not enrichment.empty:
        e = enrichment.loc[
            enrichment["configuration_id"].astype(str).eq(config_id)
            & enrichment["variable"].astype(str).eq(variable)
        ].copy()
        if not e.empty:
            st.altair_chart(charts.categorical_enrichment_bars(e), use_container_width=True)
            st.dataframe(e, hide_index=True, use_container_width=True, height=340)
    elif family == "continuous" and assignment_meta is not None and variable in assignment_meta.columns:
        d = assignment_meta.loc[assignment_meta["configuration_id"].astype(str).eq(config_id), ["cluster", variable]].copy()
        d[variable] = pd.to_numeric(d[variable], errors="coerce")
        d = d.dropna()
        if not d.empty:
            st.altair_chart(charts.continuous_boxplot(d, variable), use_container_width=True)
        if continuous_summary is not None and not continuous_summary.empty:
            s = continuous_summary.loc[
                continuous_summary["configuration_id"].astype(str).eq(config_id)
                & continuous_summary["variable"].astype(str).eq(variable)
            ]
            if not s.empty:
                st.dataframe(s, hide_index=True, use_container_width=True)

    with st.popover("ⓘ Statistical interpretation"):
        st.markdown(
            """
            The cluster solution is fixed **before** this page tests metadata. For categorical variables, 2×2 tables use Fisher's exact test; larger tables use Pearson chi-square when expected counts are adequate and permutation chi-square when they are sparse. Effect size is bias-corrected Cramér's V. Continuous variables use Kruskal–Wallis with epsilon-squared effect size.

            `q_value_scope` is the most useful correction here because it controls FDR separately within the **primary** and **sensitivity** analysis scopes. This prevents the large exploratory parameter sweep from changing the multiple-testing burden of the canonical primary analysis.
            """
        )