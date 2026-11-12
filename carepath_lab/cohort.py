"""Apply the documented first-baseline cohort rule to a derived DataFrame."""

import pandas as pd


DEATH_RELATED_DISCHARGE_CODES = {"11", "19", "20", "21"}


def select_analysis_cohort(data: pd.DataFrame) -> pd.DataFrame:
    """Exclude recorded death-related discharge codes without changing the source."""
    return data.loc[
        ~data["discharge_disposition_id"].isin(DEATH_RELATED_DISCHARGE_CODES)
    ].copy()
