"""Tools for analysing bond market reactions to yield curve and policy rate changes.

This module provides functionality to:
- Visualise yield curve spreads (US 10Y-2Y and UK 10Y-3M)
- Perform OLS regression of bond ETF returns on spread and rate changes
- Compare cumulative total returns of US Aggregate and UK Gilts ETFs
"""

from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm


class DataAnalysis:
    """Class for analysing bond market data with focus on yield curve spreads
    and monetary policy rate changes.

    Attributes
    ----------
        data (pd.DataFrame): Time series data with Date index and financial metrics
    """

    def __init__(self, data: Path) -> None:
        """Initialise DataAnalysis with data from CSV file.

        Arguments
        ---------
            data_path: Path to the CSV file containing the combined dataset
            with Date index.

        """
        self.data = pd.read_csv(data, parse_dates=["Date"], index_col="Date")

    def visualise_yield_curves(self) -> None:
        """Create a two-panel Plotly figure showing US and UK yield curve spreads.

        Left panel:  US 10Y-2Y Treasury yield spread
        Right panel: UK 10Y Gilt minus 3M interbank rate spread
        """
        data = self.data.copy()
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("US Yield Curve", "UK Yield Curve (10Y-3M Spread)"),
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["10Y-2Y Treasury Yield Spread"],
                name="US 10Y-2Y Spread",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["UK Bond Yield Spread"], name="UK 10Y-3M Spread"
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)

        fig.update_yaxes(
            title_text="US 10-Year minus 2-Year Treasury Yield Spread (%)", row=1, col=1
        )
        fig.update_yaxes(
            title_text="UK 10-Year Gilt minus 3-Month Interbank Yield Spread (%)",
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text="Yield Curve Spreads Over Time",
            height=600,
            hovermode="x unified",
            showlegend=True,
        )
        fig.show()

    def market_analysis(
        self,
    ) -> tuple[
        pd.DataFrame,
        sm.regression.linear_model.RegressionResultsWrapper,
        sm.regression.linear_model.RegressionResultsWrapper,
    ]:
        """Perform OLS regression of bond ETF returns on changes in spreads and policy rates.

        Runs two separate regressions:
          - US AGG ETF daily returns
          - UK IGLT.L ETF daily returns

        Explanatory variables:
          - Change in US 10Y-2Y spread
          - Change in UK 10Y-3M spread
          - Change in Federal Funds Rate
          - Change in UK 3-month interbank rate (IR3TIB01GBM156N)

        Returns
        -------
            data (pd.DataFrame): DataFrame with computed returns and changes
            model_us (RegressionResultsWrapper): Fitted OLS model for US bond returns
            model_uk (RegressionResultsWrapper): Fitted OLS model for UK bond returns
        """
        data = self.data.copy()
        # Compute bond returns
        data["US_Bond_Returns"] = data["Close_AGG"].pct_change()
        data["UK_Bond_Returns"] = data["Close_IGLT_L"].pct_change()

        data["US_Spread_Change"] = data["10Y-2Y Treasury Yield Spread"].diff()
        data["UK_Spread_Change"] = data["UK Bond Yield Spread"].diff()

        data["US_Rate_Change"] = data["Federal Funds Rate"].diff()
        data["UK_Rate_Change"] = data["IR3TIB01GBM156N"].diff()

        # Drop rows with NaN values to avoid errors in OLS
        data = data.dropna()

        # Run OLS regression to model policy impacts
        X = sm.add_constant(
            data[
                [
                    "US_Spread_Change",
                    "UK_Spread_Change",
                    "US_Rate_Change",
                    "UK_Rate_Change",
                ]
            ]
        )

        y_us = data["US_Bond_Returns"]
        y_uk = data["UK_Bond_Returns"]

        model_us = sm.OLS(y_us, X).fit()

        model_uk = sm.OLS(y_uk, X).fit()

        return data, model_us, model_uk

    def compare_bond_reactions(self) -> None:
        """Plot cumulative total returns of US Aggregate and UK Gilts ETFs.

        Displays an interactive Plotly figure comparing the performance
        of AGG (US) and IGLT.L (UK) over time.
        """
        data = self.data.copy()
        # Compute bond returns
        data["US_Bond_Returns"] = data["Close_AGG"].pct_change()
        data["UK_Bond_Returns"] = data["Close_IGLT_L"].pct_change()

        data["US_Cum_Return"] = (1 + data["US_Bond_Returns"].fillna(0)).cumprod() - 1
        data["UK_Cum_Return"] = (1 + data["UK_Bond_Returns"].fillna(0)).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["US_Cum_Return"], name="US AGG (cum return)"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["UK_Cum_Return"], name="UK IGLT.L (cum return)"
            )
        )
        fig.update_layout(
            title="Cumulative Returns: US Aggregate vs UK Gilts ETF",
            yaxis_title="Cumulative Total Return",
            xaxis_title="Date",
            hovermode="x unified",
            height=600,
        )
        fig.show()


def main() -> None:
    """Entry point of the script.

    Loads combined financial data from the repository's data directory
    and runs all available analyses.
    """
    repo_root = Path(__file__).parents[2].resolve()
    data_path = repo_root / "data/policy_impacts/combined_data.csv"
    analysis = DataAnalysis(data_path)
    analysis.visualise_yield_curves()
    _, _, _ = analysis.market_analysis()
    analysis.compare_bond_reactions()


if __name__ == "__main__":
    main()
