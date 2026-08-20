from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import streamlit as st


@contextmanager
def info_popover(title: str) -> Iterator[None]:
    """Render a reusable compact information popover."""

    with st.popover(f"ⓘ {title}"):
        yield


def coming_soon(project_number: int, *, subtitle: str | None = None) -> None:
    """Render a consistent placeholder for future TCGA-STAD mini projects."""

    st.title(f"TCGA-STAD Project {project_number}")
    if subtitle:
        st.caption(subtitle)
    st.info("Coming soon.", icon="🚧")
