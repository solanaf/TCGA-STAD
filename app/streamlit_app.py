from __future__ import annotations

import streamlit as st

from projects import (
    project1_expression_structure,
    project2,
    project3,
    project4,
    project5,
)

from styles import CUSTOM_CSS

st.set_page_config(
    page_title="TCGA-STAD Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True,)

pages = {
    "TCGA-STAD Projects": [
        st.Page(
            project1_expression_structure.render,
            title="1 · Gene Expression Subgroups",
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
            icon="💊",
            url_path="project-5",
        ),
    ]
}

page = st.navigation(pages, position="sidebar")

page.run()

with st.sidebar:
    st.space("large")

    st.caption("Connect")

    st.link_button(
        "GitHub",
        "https://github.com/solanaf",
        icon=":material/code:",
        width="stretch",
    )

    st.link_button(
        "LinkedIn",
        "https://www.linkedin.com/in/solanaf",
        icon=":material/work:",
        width="stretch",
    )

    st.link_button(
        "Portfolio",
        "https://solanaf.github.io/portfolio-v1/#portfolio",
        icon=":material/science:",
        width="stretch",
    )

    st.link_button(
        "Personal website",
        "https://solanaf.github.io/portfolio-v1/",
        icon=":material/language:",
        width="stretch",
    )

    st.divider()

    st.markdown(
        """
        **Solana Fernandez**  
        [sbfernandez@ucsd.edu](mailto:sbfernandez@ucsd.edu)
        """
    )