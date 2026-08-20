from __future__ import annotations

from utils.components import coming_soon


from pathlib import Path

import pandas as pd
import streamlit as st

STEPS = (
    "Overview",
)

def _sidebar() -> tuple[str, Path]:
    """Render Project 1 workflow navigation and use the launch directory as root."""

    step = st.sidebar.radio(
        "Workflow step",
        STEPS,
        key="project2_step",
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

def render() -> None:
    step, root = _sidebar()

    st.title("TCGA-STAD · Mutational Cancer Driver Genes")
    st.caption(
        "Project 2 · coming soon ..."
    )
    st.divider()

    if step == "Overview":
        render_overview(root)



def render_overview(root: Path) -> None:
    st.subheader(
        "Overview",
    )
    st.write("""
This exploration is inspired by **_A compendium of mutational cancerdriver genes (2021). Francisco Martínez-Jiménez et al._**
    
This page is a cache-backed gene expression explorer of the TCGA-STAD dataset. Use the workflow selector in the sidebar to inspect what happened at each stage and switch among the parameter combinations that were already computed.
    """
    )