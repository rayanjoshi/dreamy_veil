"""Module for analysing capital expenditure trends of the Magnificent Seven companies.

This module provides tools to read combined financial data, construct a quarterly panel
dataset, perform panel regression analysis, and create visualisations of capex growth.
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
import plotly.express as px


class DataAnalysis:
    """Class for performing data analysis on Magnificent Seven corporate data.

    Handles loading of combined data, construction of quarterly panel datasets,
    regression modelling and visualisation of key financial metrics.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialise the DataAnalysis object with a data directory.

        Arguments
        ---------
            data_dir (Path): Directory containing the combined_data.csv file
        """
        self.data_dir = data_dir
        self.data = pd.read_csv(
            self.data_dir / "combined_data.csv", index_col=0, parse_dates=True
        )

    def build_panel(self) -> pd.DataFrame:
        """Construct a quarterly panel dataset for the Magnificent Seven companies.

        Resamples data to quarterly frequency, computes growth rates and differences,
        builds a panel structure with ticker and date as multi-index, and saves
        the result to CSV.

        Returns
        -------
            pd.DataFrame: Quarterly panel with multi-index (Ticker, Date)
        """
        data = self.data.copy()
        data = (
            data.resample("QE")
            .agg(
                {
                    "FedFunds": "last",
                    "GDP": "last",
                    **{
                        col: "last"
                        for col in data.columns
                        if "close" in col or "capex" in col
                    },
                }
            )
            .dropna(how="all")
        )

        # Compute changes/growth
        data["Delta_FedFunds"] = data["FedFunds"].diff()
        data["GDP_Growth"] = data["GDP"].pct_change(fill_method=None)

        magnificent_seven = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        panel_list = []

        for stock in magnificent_seven:
            df = pd.DataFrame(
                {
                    "Ticker": stock,
                    "Date": data.index,
                    "FedFunds": data["FedFunds"],
                    "Delta_FedFunds": data["Delta_FedFunds"],
                    "GDP_Growth": data["GDP_Growth"],
                    "Close": data.get(f"{stock}_close"),
                    "Capex": data.get(f"{stock}_capex"),
                    "Assets": data.get(f"{stock}_assets", pd.Series()),
                }
            )
            df["Return"] = df["Close"].pct_change(fill_method=None)
            df["Capex_Growth"] = df["Capex"].pct_change(fill_method=None)
            panel_list.append(df.set_index(["Ticker", "Date"]))

        panel = (
            pd.concat(panel_list)
            .sort_index()
            .dropna(subset=["Capex_Growth", "Delta_FedFunds"])
        )
        panel.to_csv(self.data_dir / "mag7_panel_quarterly.csv")
        return panel

    def panel_regression(self, panel: pd.DataFrame) -> RegressionResultsWrapper:
        """Estimate panel regression of capex growth on macroeconomic and firm variables.

        Fits an OLS model with ticker fixed effects using statsmodels formula API.
        Standard errors are clustered by ticker.

        Arguments
        ---------
            panel (pd.DataFrame): Quarterly panel dataset with multi-index

        Returns
        -------
            RegressionResultsWrapper: Fitted model results
        """

        panel = panel.reset_index()

        reg_vars = [
            "Capex_Growth",
            "Delta_FedFunds",
            "GDP_Growth",
            "Return",
            "Ticker",
        ]

        panel = panel.dropna(subset=reg_vars).copy()
        panel["Date"] = pd.to_datetime(panel["Date"])

        model = smf.ols(
            "Capex_Growth ~ Delta_FedFunds + GDP_Growth + Return + C(Ticker)",
            data=panel,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["Ticker"]},
        )

        print(model.summary())

        return model

    def visualise_panel(self, panel: pd.DataFrame) -> None:
        """Create line plot visualisation of quarterly capex growth rates.

        Displays a faceted line chart for each Magnificent Seven company,
        showing capex growth from 2022 onwards.

        Arguments
        ---------
            panel (pd.DataFrame): Quarterly panel dataset with multi-index
        """
        panel = panel.loc["2022-01-01":]
        fig = px.line(
            panel.reset_index(),
            x="Date",
            y="Capex_Growth",
            color="Ticker",
            facet_col="Ticker",
            facet_col_wrap=3,
            title="Mag 7 Capex Growth Rates (Quarterly)",
        )
        fig.show()


def main() -> None:
    """Entry point for the Magnificent Seven capex analysis script."""
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "corporate_decisions"
    analysis = DataAnalysis(data_dir)
    data = analysis.build_panel()
    analysis.visualise_panel(data)


if __name__ == "__main__":
    main()
