from __future__ import annotations

import streamlit as st

from projects import (
    project1_expression_structure,
    project2,
    project3,
    project4,
    project5,
)


st.set_page_config(
    page_title="TCGA-STAD Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, .18);
        padding: .8rem;
        border-radius: .7rem;
    }
    .small-note {
        opacity: .72;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "TCGA-STAD mini projects": [
        st.Page(
            project1_expression_structure.render,
            title="1 · Expression structure",
            icon="🧬",
            default=True,
        ),
        st.Page(
            project2.render,
            title="2 · Coming soon",
            icon="🧪",
            url_path="project-2",
        ),
        st.Page(
            project3.render,
            title="3 · Coming soon",
            icon="🔬",
            url_path="project-3",
        ),
        st.Page(
            project4.render,
            title="4 · Coming soon",
            icon="📊",
            url_path="project-4",
        ),
        st.Page(
            project5.render,
            title="5 · Coming soon",
            icon="🧠",
            url_path="project-5",
        ),
    ]
}

page = st.navigation(pages, position="sidebar")
page.run()
