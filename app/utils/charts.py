from __future__ import annotations

import altair as alt
import pandas as pd


def scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str | None = None,
    tooltip_extra: list[str] | None = None,
) -> alt.Chart:
    tooltip = [c for c in ["sample_id", *(tooltip_extra or [])] if c in data.columns]
    encoding: dict[str, object] = {
        "x": alt.X(x, title=x),
        "y": alt.Y(y, title=y),
        "tooltip": tooltip,
    }
    if color and color in data.columns:
        if pd.api.types.is_numeric_dtype(data[color]) and data[color].nunique(dropna=True) > 12:
            encoding["color"] = alt.Color(color, type="quantitative", title=color)
        else:
            encoding["color"] = alt.Color(color, type="nominal", title=color)
    return (
        alt.Chart(data)
        .mark_circle(size=58, opacity=0.78)
        .encode(**encoding)
        .properties(title=title, height=560)
        .interactive()
    )


def line(data: pd.DataFrame, x: str, y: str, *, title: str, highlight_x: float | int | None = None) -> alt.Chart:
    base = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(x=alt.X(x, title=x), y=alt.Y(y, title=y), tooltip=[x, y])
        .properties(title=title, height=330)
    )
    if highlight_x is None:
        return base
    rule = alt.Chart(pd.DataFrame({x: [highlight_x]})).mark_rule(strokeDash=[5, 5]).encode(x=x)
    return base + rule


def categorical_enrichment_bars(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("cluster:N", title="Cluster"),
            y=alt.Y("within_cluster_percent:Q", title="Within-cluster (%)", stack="zero"),
            color=alt.Color("category:N", title="Category"),
            tooltip=["cluster:N", "category:N", "observed:Q", "expected:Q", "pearson_residual:Q"],
        )
        .properties(height=420)
    )


def continuous_boxplot(data: pd.DataFrame, variable: str) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_boxplot(size=42)
        .encode(
            x=alt.X("cluster:N", title="Cluster"),
            y=alt.Y(f"{variable}:Q", title=variable),
            color=alt.Color("cluster:N", legend=None),
        )
        .properties(height=430)
    )
