"""Module for identifying and visualising S&P 500 reactions to FOMC monetary policy shocks.

This module provides a class to detect significant interest rate changes (shocks)
around FOMC announcement dates, compute cumulative returns, perform a simple
lagged-return regression, and visualise average market reactions.
"""
import pandas as pd
from typing import Tuple
import statsmodels.api as sm
import plotly.express as px
from pathlib import Path


class IdentifyShockEvents:
    """Class for detecting and analysing S&P 500 responses to FOMC rate shocks.

    Attributes
    ----------
        data (pd.DataFrame): Input DataFrame with Date index and columns including
            'SP500_Return' and 'Rate_Change'.
        _fomc_dates (pd.DatetimeIndex): Hard-coded list of FOMC announcement dates.
    """
    def __init__(self, data):
        """Initialise the shock identification with market and rate data.

        Arguments
        ----------
            data: DataFrame indexed by Date containing at least 'SP500_Return'
                (daily decimal returns) and 'Rate_Change' (daily change in
                effective federal funds rate as decimal).
        """
        self.data = data
        fomc_dates = pd.to_datetime(
            [
                "2020-01-29",
                "2020-03-03",
                "2020-03-15",
                "2020-04-29",
                "2020-06-10",
                "2020-07-29",
                "2020-09-16",
                "2020-11-05",
                "2020-12-16",
                "2021-01-27",
                "2021-03-17",
                "2021-04-28",
                "2021-06-16",
                "2021-07-28",
                "2021-09-22",
                "2021-11-03",
                "2021-12-15",
                "2022-01-26",
                "2022-03-16",
                "2022-05-04",
                "2022-06-15",
                "2022-07-27",
                "2022-09-21",
                "2022-11-02",
                "2022-12-14",
                "2023-02-01",
                "2023-03-22",
                "2023-05-03",
                "2023-06-14",
                "2023-07-26",
                "2023-09-20",
                "2023-11-01",
                "2023-12-13",
                "2024-01-31",
                "2024-03-20",
                "2024-05-01",
                "2024-06-12",
                "2024-07-31",
                "2024-09-18",
                "2024-11-07",
                "2024-12-18",
                "2025-01-29",
                "2025-03-19",
                "2025-05-07",
                "2025-06-18",
                "2025-07-30",
                "2025-09-17",
                "2025-10-29",
                "2025-12-10",
            ]
        ).ravel()
        self._detect_shocks(fomc_dates, threshold=10)

    def identify_shock_events(self) -> None:
        """Run the full pipeline: detect shocks, fit a simple model, and visualise results."""
        df, shock_events = self._detect_shocks(self.data.index, threshold=10)
        self._fit_shocks(df)
        self._visualise_shocks(shock_events)

    def _detect_shocks(
        self, dates: pd.Series, threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect rate shocks and build event windows around FOMC dates.

        Arguments
        ----------
            dates: DatetimeIndex of FOMC announcement dates.
            threshold: Absolute rate change in basis points that qualifies as a shock.

        Returns
        --------
            Tuple containing:
                - df: Original data enriched with shock classification columns.
                - events_df: Concatenated event windows (±10 to +20 days) around each FOMC date.
        """
        df = self.data.copy()

        df["is_fomc_date"] = df.index.isin(dates)
        df["FOMC_Window"] = df.index.map(
            lambda d: next((i for i, date in enumerate(dates) if d >= date), None)
        )
        df["Rate_Change_bp"] = df["Rate_Change"] * 100

        df["Shock_Type"] = "No Shock"
        mask = df["Rate_Change_bp"].abs() >= threshold
        df.loc[mask & (df["Rate_Change_bp"] > 0), "Shock_Type"] = "Hike"
        df.loc[mask & (df["Rate_Change_bp"] < 0), "Shock_Type"] = "Cut"

        event_windows = []
        for event_date in dates:
            window = df.loc[
                event_date - pd.Timedelta(days=10) : event_date + pd.Timedelta(days=20)
            ].copy()
            if not window.empty:
                window["Days_From_Event"] = (window.index - event_date).days
                window["Event_Date"] = event_date

                window["SP500_Return_%"] = window["SP500_Return"] * 100

                cum_factor = (1 + window["SP500_Return"]).cumprod()
                window["Cum_Return_%"] = (cum_factor - 1) * 100

                event_windows.append(
                    window[
                        [
                            "SP500_Return_%",
                            "Days_From_Event",
                            "Event_Date",
                            "Cum_Return_%",
                            "Shock_Type",
                        ]
                    ]
                )

        if event_windows:
            events_df = pd.concat(event_windows)
        else:
            events_df = df.iloc[0:0][
                [
                    "SP500_Return_%",
                    "Days_From_Event",
                    "Event_Date",
                    "Cum_Return_%",
                    "Shock_Type",
                ]
            ]

        return df, events_df

    def _fit_shocks(self, data: pd.DataFrame) -> None:
        """Fit a simple OLS regression of current returns on lagged returns.

        Prints the statsmodels summary to console.

        Arguments
        ----------
            data: DataFrame containing at least 'SP500_Return'.
        """
        df_reg = data.copy()
        df_reg["lagged_return"] = df_reg["SP500_Return"].shift(1)
        df_reg.dropna(inplace=True)

        X = sm.add_constant(df_reg["lagged_return"])
        y = df_reg["SP500_Return"]
        model = sm.OLS(y, X).fit()
        print(model.summary())

    def _visualise_shocks(self, data: pd.DataFrame) -> None:
        """Create Plotly line charts of average cumulative S&P 500 returns around FOMC events.

        Shows two figures:
            1. Overall average cumulative return.
            2. Average cumulative return separated by shock type (Hike/Cut/No Shock).

        Arguments
        ----------
            data: Event window DataFrame produced by _detect_shocks.
        """
        if data.empty:
            print("No event data to visualise.")
            return

        avg_cum = data.groupby("Days_From_Event")["Cum_Return_%"].mean().reset_index()

        fig = px.line(
            avg_cum,
            x="Days_From_Event",
            y="Cum_Return_%",
            title="Average S&P 500 Cumulative Return (%) Around FOMC Announcements (2020–2025)",
        )
        fig.add_vline(x=0, line_dash="dash", annotation_text="FOMC Announcement")
        fig.update_layout(
            yaxis_title="Cumulative Return (%)", xaxis_title="Days From Event"
        )
        fig.show()

        avg_by_type = (
            data.groupby(["Days_From_Event", "Shock_Type"])["Cum_Return_%"]
            .mean()
            .reset_index()
        )

        fig_type = px.line(
            avg_by_type,
            x="Days_From_Event",
            y="Cum_Return_%",
            color="Shock_Type",
            title="Average S&P 500 Cumulative Return (%) by Shock Type (2020–2025)",
        )
        fig_type.add_vline(x=0, line_dash="dash", annotation_text="FOMC Announcement")
        fig_type.update_layout(
            yaxis_title="Cumulative Return (%)", xaxis_title="Days From Event"
        )
        fig_type.show()


def main():
    """Load combined data and execute the full shock identification and visualisation pipeline."""
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "shocks_and_reactions" / "combined_data.csv"

    data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    shock_identifier = IdentifyShockEvents(data)
    shock_identifier.identify_shock_events()


if __name__ == "__main__":
    main()
