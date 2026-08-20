from __future__ import annotations
import streamlit as st

DARK_PURPLE = "#8250D8"
LIGHT_PURPLE = "#AA87E6"
OFF_BLACK = "#2D3142CF"
LIGHT_BLUE = "#43ADFFC6"
MID_BLUE = "#1C89DCF0"
DARK_BLUE = "#285DC7"

PLOT_TITLE_COLOR = MID_BLUE

PLOT_TITLE_SIZE = 36

H4_WEIGHT = 600


CUSTOM_CSS = f"""
<style>
[data-testid="stHeadingWithActionElements"] h1> span:first-child {{
    font-size: 44px;
    font-weight: 1000;
    color: {DARK_PURPLE}
}}
h3 > span:first-child {{
    font-size: 38px;
    font-weight: 750;
    color: {DARK_BLUE};
}}
h4 > span:first-child {{
    font-size: 26px;
    font-weight: {H4_WEIGHT};
    color: {MID_BLUE};
}}
div[data-testid="stSelectbox"]
label[data-testid="stWidgetLabel"]
div[data-testid="stMarkdownContainer"] p {{
    font-size: 20px;
    font-weight: 500;
    color: {LIGHT_BLUE};
}}
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {{
    font-size: 20px;
    font-weight: 500;
    color: {LIGHT_BLUE};
}}
[data-testid="stRadio"]:has(
    [data-testid="stRadioGroup"][aria-label="Metadata type"]
) [data-testid="stWidgetLabel"] p {{
    font-size: 20px;
    font-weight: 500;
    color: {LIGHT_BLUE};
}}
[data-testid="stNavSectionHeader"]
[data-testid="stMarkdownContainer"] p {{
    font-size: 20px;
    font-weight: 600;
    color: {DARK_PURPLE};
}}
[data-testid="stSidebarNavLink"] [data-testid="stMarkdownContainer"] p {{
    font-size: 18px;
    font-weight: 400;
    color: {LIGHT_PURPLE};
}}
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{
    font-size: 20px;
    font-weight: 600;
    color: {MID_BLUE};
}}
[data-testid="stRadio"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p {{
    font-size: 18px;
    font-weight: 400;
    color: {LIGHT_BLUE};
}}
[data-testid="stMetric"] {{
    border: 1px solid rgba(128, 128, 128, .18);
    padding: .8rem;
    border-radius: .7rem;
    
}}
[data-testid="stCaptionContainer"] p {{
    font-size: 20px;
    font-weight: 2000;
    color: {OFF_BLACK};
}}

</style>
"""