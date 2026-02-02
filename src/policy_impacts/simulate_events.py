"""Event simulation module for modelling policy impacts on bond returns.

This module provides functionality to:
- Load historical market data
- Fit simple predictive models for US and UK bond returns
- Simulate future bond return paths under hypothetical rate change scenarios
- Visualise cumulative return paths and spread evolution
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parents[1].resolve()))
from policy_impacts.data_analysis import DataAnalysis


class EventSimulator:
    """Simulator for assessing policy rate change impacts on bond indices.

    Fits linear models on historical data and uses them to forecast
    bond returns under user-defined interest rate change scenarios.

    Attributes
    -----------
        data_path (Path): Path to the input CSV data file
        data (pd.DataFrame): Historical market data (loaded on demand)
        us_model: Fitted model for US bond returns
        uk_model: Fitted model for UK bond returns
        last_date (pd.Timestamp): Most recent date in the training data
    """
    def __init__(self, data: Path) -> None:
        """Initialize the event simulator with data location.

        Arguments
        ----------
            data: Path to the CSV file containing market data
        """
        self.data_path = data
        self.data = pd.DataFrame()
        self.us_model = None
        self.uk_model = None
        self.last_date = pd.to_datetime("2020-01-01")

    def fit_model(self) -> None:
        """Load data and train predictive models for US and UK bond returns.

        Calls the market_analysis() method from DataAnalysis to obtain
        preprocessed data and fitted models.

        Raises
        ------
            Whatever exceptions DataAnalysis.market_analysis() may raise
        """
        self.data, self.us_model, self.uk_model = DataAnalysis(
            self.data_path
        ).market_analysis()

    def simulate_event(
        self,
        name: str,
        total_us_rate_change: float,
        total_uk_rate_change: float,
        months: int = 12,
        spread_impact_factor: float = 0.5,
    ) -> pd.DataFrame:
        """Simulate bond market impact of a monetary policy event.

        Creates a forward-looking simulation of cumulative bond returns
        and yield spread evolution based on assumed rate changes.

        Arguments
        ----------
            name: Descriptive name of the simulated scenario
            total_us_rate_change: Total change in Fed Funds Rate (decimal)
            total_uk_rate_change: Total change in UK 3M interbank rate (decimal)
            months: Number of months over which the change is applied
            spread_impact_factor: Multiplier determining spread sensitivity
                                 to policy rate changes (0.0–1.0 typical)

        Returns
        -------
            DataFrame containing monthly simulation results with columns:
                - US_Rate_Change_Monthly
                - UK_Rate_Change_Monthly
                - US_Spread_Change
                - UK_Spread_Change
                - Predicted_US_Return
                - Predicted_UK_Return
                - Cum_US_Return
                - Cum_UK_Return
                - US_Spread_Level
                - UK_Spread_Level

        Raises
        ------
            ValueError: If models have not been fitted yet
        """
        if self.us_model is None or self.uk_model is None:
            raise ValueError(
                "Models must be fitted before simulating events. Call fit_model() first."
            )

        sim_index = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1), periods=months, freq="ME"
        )
        sim_data = pd.DataFrame(index=sim_index)

        sim_data["Month"] = range(1, months + 1)

        monthly_us = total_us_rate_change / months
        monthly_uk = total_uk_rate_change / months

        current_us_spread = self.data["10Y-2Y Treasury Yield Spread"].iloc[-1]
        current_uk_spread = self.data["UK Bond Yield Spread"].iloc[-1]
        cum_us_ret = 0.0
        cum_uk_ret = 0.0

        results = []

        for i, _ in enumerate(sim_data.index):
            us_spread_chg = spread_impact_factor * monthly_us
            uk_spread_chg = spread_impact_factor * monthly_uk

            X_new = np.array(
                [[1, us_spread_chg, uk_spread_chg, monthly_us, monthly_uk]]
            )

            prediction_us_ret = self.us_model.predict(X_new)[0]
            prediction_uk_ret = self.uk_model.predict(X_new)[0]

            cum_us_ret += prediction_us_ret
            cum_uk_ret += prediction_uk_ret
            current_us_spread += us_spread_chg
            current_uk_spread += uk_spread_chg

            results.append(
                {
                    "Month": sim_data["Month"].iloc[i],
                    "US_Rate_Change_Monthly": monthly_us,
                    "UK_Rate_Change_Monthly": monthly_uk,
                    "US_Spread_Change": us_spread_chg,
                    "UK_Spread_Change": uk_spread_chg,
                    "Predicted_US_Return": prediction_us_ret,
                    "Predicted_UK_Return": prediction_uk_ret,
                    "Cum_US_Return": cum_us_ret,
                    "Cum_UK_Return": cum_uk_ret,
                    "US_Spread_Level": current_us_spread,
                    "UK_Spread_Level": current_uk_spread,
                }
            )

        sim_results = pd.DataFrame(results).set_index("Month")

        print(f"\nScenario: {name}")
        print(f"Total US rate change: {total_us_rate_change * 100:.1f} bp")
        print(f"Total UK rate change: {total_uk_rate_change * 100:.1f} bp")
        print(f"Over {months} months")
        print(f"Final cum US bond return (approx): {cum_us_ret * 100:.2f}%")
        print(f"Final cum UK bond return (approx): {cum_uk_ret * 100:.2f}%")
        print(f"US spread change: {sim_results['US_Spread_Change'].sum() * 100:.1f} bp")
        print(f"UK spread change: {sim_results['UK_Spread_Change'].sum() * 100:.1f} bp")

        return sim_results

    def plot_simulation(self, sim_df: pd.DataFrame, title: str) -> None:
        """Create an interactive Plotly visualisation of simulation results.

        Plots cumulative returns for US and UK bond indices (left axis)
        and yield spread levels (right axis).

        Arguments
        ----------
            sim_df: DataFrame returned by simulate_event()
            title: Title of the plot
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df["Cum_US_Return"],
                name="US AGG Cum Return",
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df["Cum_UK_Return"],
                name="UK IGLT.L Cum Return",
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sim_df.index,
                y=sim_df["US_Spread_Level"],
                name="US 10Y-2Y Spread",
                mode="lines",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Cumulative Return",
            yaxis2=dict(title="Spread Level (%)", overlaying="y", side="right"),
            hovermode="x unified",
            height=600,
        )
        fig.show()


def main() -> None:
    """Run example simulation demonstrating QE-type policy impact."""
    repo_root = Path(__file__).parents[2].resolve()
    data_path = repo_root / "data/policy_impacts/combined_data.csv"
    simulator = EventSimulator(data_path)
    simulator.fit_model()

    sim_df = simulator.simulate_event(
        name="QE - 50bp cut (low growth)",
        total_us_rate_change=-0.5,
        total_uk_rate_change=-0.3,
        months=12,
        spread_impact_factor=0.0,
    )

    simulator.plot_simulation(
        sim_df, title="Simulated Bond Returns under QE Policy Event"
    )


if __name__ == "__main__":
    main()
