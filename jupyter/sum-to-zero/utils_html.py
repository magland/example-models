# json to HTML
import json
import pandas as pd
from IPython.display import HTML
import numpy as np
import pandas as pd
from typing import List


def extract_numeric_index(idx: pd.Index) -> pd.Series:
    return idx.str.extract(r'[a-z_]*\[(\d+)\]', expand=False).astype(int)

# add dividers every nth row
from pandas.io.formats.style import Styler
def style_dataframe(df: pd.DataFrame, modulus: int) -> Styler:
    def highlight_every_nth_row(row: pd.Series, row_index: int, modulus: int) -> list[str]:
        if (row_index + 1) % modulus == 0:  # add border
            return ['border-bottom: 3px double black'] * len(row)
        return [''] * len(row)
    return (df.style
              .apply(lambda row: highlight_every_nth_row(row, df.index.get_loc(row.name), modulus), axis=1)
              .format(precision=2)
           )


# Filter and sort predictors
def summarize_predictor(df: pd.DataFrame, name: str) -> pd.DataFrame:
    pred_summary = df.filter(regex=name, axis=0).sort_index()
    if "[" in name:
        pred_summary = pred_summary.sort_index(key=extract_numeric_index)
    
    return pred_summary[['Mean', 'StdDev', 'ESS_bulk/s', 'R_hat']]


# side-by-side tables
from IPython.core.display import display, HTML
def display_side_by_side(
    html_left: str,
    html_right: str,
    labels: str = "",
    title_left: str = "Small Dataset",
    title_right: str = "Large Dataset"
) -> None:
    """
    Displays two HTML tables side by side in a Jupyter Notebook.
    """
    html_code = f"""
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <div style="width: 10%; border: 1px solid #ddd; padding: 3px;">
            <b><i>&nbsp;</i></b>
            {labels}
        </div>
        <div style="width: 45%; border: 1px solid #ddd; padding: 5px;">
            <b><i>{title_left}</i></b>
            {html_left}
        </div>
        <div style="width: 45%; border: 1px solid #ddd; padding: 5px;">
            <b><i>{title_right}</i></b>
            {html_right}
        </div>
    </div>
    """
    display(HTML(html_code))

def expand_true(datagen_values: List[float], header: str) -> str:
    # Create HTML for datagen values column
    datagen_values_html = f"<table border='1' class='dataframe'><thead><tr><th>{header}</th></tr></thead><tbody>"
    
    # Add each datagen value three times (one for each model)
    for val in datagen_values:
        # Show value for first row of each group
        datagen_values_html += f"<tr><td>{val:.2f}</td></tr>"
        # Empty cells for the other two rows
        datagen_values_html += "<tr><td>&nbsp;</td></tr>"
        datagen_values_html += "<tr><td>&nbsp;</td></tr>"
    datagen_values_html += "</tbody></table>"
    return datagen_values_html
