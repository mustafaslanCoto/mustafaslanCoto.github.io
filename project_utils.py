import pandas as pd
from IPython.display import display
import pandas as pd

def table_styler(
    df,
    table_header = [('background-color', '#20808D'), ('text-align', 'left'),
                    ('color', '#FBFAF4'), ('font-size', '12px')],
    cells_format = [('font-size', '12px'), ('text-align', 'left')],
    highlight_cells = None,
    highlighted_cell_style = 'font-weight: bold; background-color: None;',
    numeric_highlights = None,
    float_format = "{:.3f}"
):
    """
    Apply styles to a DataFrame for display in Jupyter Notebook or Quarto.

    Parameters:
        df (pd.DataFrame): The DataFrame to style.
        table_header (list of tuples): CSS styles for the table header.
            Defaults is [('background-color', '#20808D'), ('text-align', 'left'),
             ('color', '#FBFAF4'), ('font-size', '12px')].
        cells_format (list of tuples): CSS styles for the table cells.
            Defaults is [('font-size', '12px'), ('text-align', 'left')].
        highlight_cells (list of (row index, col name) tuples): Cells to highlight.
            Defaults to None. Example: [(0, 'col1'), (1, 'col2')].
        highlighted_cell_style (str): CSS style for highlighted cells.
            Defaults to 'font-weight: bold; background-color: None;'.
        numeric_highlights (dict): Dict with column name as key and dict as value:
            {
                "col1": {"condition": "greater", "threshold": 10, "style": "font-weight: bold; background-color: None;"},
                "col2": {"condition": "less_equal", "threshold": 5, "style": "font-weight: bold; background-color: None;"},
                ...
            }
            condition can be 'greater', 'less', 'equal', 'not_equal', 'greater_equal', or 'less_equal'.

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame.
    """
    def combined_styler(
        df,
        highlight_cells=highlight_cells,
        highlighted_cell_style=highlighted_cell_style,
        numeric_highlights=numeric_highlights
    ):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        # Highlight specific cells
        if highlight_cells:
            # check if highlight_cells is a list of tuples
            if not isinstance(highlight_cells, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in highlight_cells):
                raise ValueError("highlight_cells must be a list of tuples with (row index, column name).")
            for row, col in highlight_cells:
                if row in df.index and col in df.columns:
                    styles.loc[row, col] += highlighted_cell_style

        # Numeric highlights
        if numeric_highlights:
            # check if numeric_highlights is a dict
            if not isinstance(numeric_highlights, dict):
                raise ValueError("numeric_highlights must be a dictionary with column names as keys.")
            for col, config in numeric_highlights.items():
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors='coerce')
                    condition = config.get("condition", "greater")
                    threshold = config.get("threshold")
                    style = config.get("style", "font-weight: bold; background-color: yellow;")
                    if threshold is not None:
                        if condition == 'greater':
                            mask = vals > threshold
                        elif condition == 'less':
                            mask = vals < threshold
                        elif condition == 'equal':
                            mask = vals == threshold
                        elif condition == 'not_equal':
                            mask = vals != threshold
                        elif condition == 'greater_equal':
                            mask = vals >= threshold
                        elif condition == 'less_equal':
                            mask = vals <= threshold
                        else:
                            raise ValueError(f"Unsupported condition: {condition}")
                        styles.loc[mask, col] += style

        return styles

    styled = (
        df.style
        .apply(
            combined_styler,
            highlight_cells=highlight_cells,
            highlighted_cell_style=highlighted_cell_style,
            numeric_highlights=numeric_highlights,
            axis=None
        )
        .set_table_styles([
            {'selector': 'table', 'props': [('width', '100%'), ('border-collapse', 'collapse')]},
            {'selector': 'th', 'props': table_header},
            {'selector': 'td', 'props': cells_format}
        ])
        .hide(axis="index")
    )

    # Set float format for all float columns to 3 decimals
    float_cols = df.select_dtypes(include=['float', 'float64']).columns
    if len(float_cols) > 0:
        styled = styled.format({col: float_format.format for col in float_cols})

    return display(styled)