"""Module for simulating the impact of monetary policy scenarios on corporate capex.

This module uses fitted panel regression coefficients to project forward-looking
capital expenditure growth under hypothetical Fed policy paths.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.corporate_decisions.data_analysis import DataAnalysis


class EventSimulator:
    """Class for simulating corporate capex responses to monetary policy scenarios.

    Uses a fitted panel regression model to project quarterly capex growth under
    different Fed rate change assumptions.
    """

    def __init__(self, data_directory: Path) -> None:
        """Initialise the EventSimulator with historical panel data.

        Arguments
        ---------
            data_directory (Path): Path to the corporate_decisions data folder
        """
        repo_root = Path(__file__).resolve().parents[2]
        self.data_directory = repo_root / data_directory

        # Load historical panel data
        panel_path = self.data_directory / "mag7_panel_quarterly.csv"
        self.data = pd.read_csv(panel_path, index_col=[0, 1], parse_dates=True)

        # Fit the panel regression model
        analysis = DataAnalysis(data_dir=self.data_directory)
        self.model = analysis.panel_regression(panel=self.data)

        # Extract coefficients for simulation
        self._extract_coefficients()

        # Get most recent values for simulation baseline
        self.last_date = self.data.reset_index()["Date"].max()
        self.magnificent_seven = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
        ]

    def _extract_coefficients(self) -> None:
        """Extract regression coefficients from the fitted model."""
        params = self.model.params

        # Store coefficients (with fallback defaults if not present)
        self.coef_delta_fedfunds = params.get("Delta_FedFunds", -0.05)
        self.coef_gdp_growth = params.get("GDP_Growth", 0.5)
        self.coef_return = params.get("Return", 0.1)
        self.intercept = params.get("Intercept", 0.02)

    def simulate_event(
        self,
        event_type: str,
        quarters: int = 4,
        rate_change_per_quarter: float = 0.0,
        gdp_growth_assumption: float = 0.005,
    ) -> pd.DataFrame:
        """Simulate forward-looking capex growth under a policy scenario.

        Arguments
        ---------
            event_type (str): Type of scenario ('tightening', 'easing', 'status_quo')
            quarters (int): Number of quarters to simulate forward
            rate_change_per_quarter (float): Fed Funds rate change per quarter (in pp, e.g., 0.25 for 25bp)
            gdp_growth_assumption (float): Assumed quarterly GDP growth rate (e.g., 0.005 for 0.5%)

        Returns
        -------
            pd.DataFrame: Simulated panel with predicted capex growth by ticker and quarter
        """
        simulation_results = []

        # Historical average return volatility for forward simulation
        hist_return_vol = self.data["Return"].std()

        # Track cumulative changes per ticker
        ticker_cumulative = {ticker: 1.0 for ticker in self.magnificent_seven}

        for q in range(1, quarters + 1):
            for ticker in self.magnificent_seven:
                # Get most recent actual values for this ticker
                ticker_data = self.data.reset_index()
                ticker_data = ticker_data[ticker_data["Ticker"] == ticker].sort_values(
                    "Date"
                )

                if ticker_data.empty:
                    continue

                # Project date forward
                future_date = pd.Timestamp(self.last_date) + pd.DateOffset(months=3 * q)

                # Simulate stock return (mean-reverting random walk with historical vol)
                # Use a simple assumption: returns decay toward zero with some noise
                np.random.seed(
                    42 + q + hash(ticker) % 1000
                )  # Reproducible but varied by ticker/quarter
                simulated_return = np.random.normal(
                    0.02, hist_return_vol
                )  # Assume mild positive drift

                # Policy shock in this quarter
                delta_fedfunds = rate_change_per_quarter
                gdp_growth = gdp_growth_assumption

                # Predict capex growth using fitted model
                predicted_capex_growth = (
                    self.intercept
                    + self.coef_delta_fedfunds * delta_fedfunds
                    + self.coef_gdp_growth * gdp_growth
                    + self.coef_return * simulated_return
                )

                # Update cumulative change (compound growth)
                ticker_cumulative[ticker] *= 1 + predicted_capex_growth
                cumulative_change = ticker_cumulative[ticker] - 1.0

                simulation_results.append(
                    {
                        "Ticker": ticker,
                        "Quarter": q,
                        "Date": future_date,
                        "Event_Type": event_type,
                        "Delta_FedFunds": delta_fedfunds,
                        "GDP_Growth": gdp_growth,
                        "Simulated_Return": simulated_return,
                        "Predicted_Capex_Growth": predicted_capex_growth,
                        "Cumulative_Capex_Change": cumulative_change,
                    }
                )

        return pd.DataFrame(simulation_results)

    def plot_simulation(
        self, simulation_df: pd.DataFrame, title: str = "Policy Scenario Simulation"
    ) -> None:
        """Create interactive visualisation of simulated capex paths.

        Arguments
        ---------
            simulation_df (pd.DataFrame): Output from simulate_event()
            title (str): Plot title
        """
        # Create subplots: one for quarterly growth, one for cumulative change
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Predicted Quarterly Capex Growth (%)",
                "Cumulative Capex Change (%)",
            ),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
        )

        # Plot each ticker
        for ticker in simulation_df["Ticker"].unique():
            ticker_data = simulation_df[simulation_df["Ticker"] == ticker].sort_values(
                "Quarter"
            )

            # Quarterly growth
            fig.add_trace(
                go.Scatter(
                    x=ticker_data["Quarter"],
                    y=ticker_data["Predicted_Capex_Growth"] * 100,
                    name=ticker,
                    mode="lines+markers",
                    legendgroup=ticker,
                ),
                row=1,
                col=1,
            )

            # Cumulative change
            fig.add_trace(
                go.Scatter(
                    x=ticker_data["Quarter"],
                    y=ticker_data["Cumulative_Capex_Change"] * 100,
                    name=ticker,
                    mode="lines+markers",
                    showlegend=False,
                    legendgroup=ticker,
                ),
                row=2,
                col=1,
            )

        # Add horizontal reference lines
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1
        )

        # Update axes
        fig.update_xaxes(title_text="Quarter Ahead", row=1, col=1)
        fig.update_xaxes(title_text="Quarter Ahead", row=2, col=1)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Change (%)", row=2, col=1)

        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.show()


def main() -> None:
    """Entry point for testing simulation functionality."""
    data_directory = Path("data/corporate_decisions")
    simulator = EventSimulator(data_directory=data_directory)

    print("\n--- Scenario 1: Tightening Cycle ---")
    tightening = simulator.simulate_event(
        event_type="tightening",
        quarters=4,
        rate_change_per_quarter=0.25,
        gdp_growth_assumption=0.005,
    )
    simulator.plot_simulation(tightening, title="Tightening Cycle: 100bp Hikes")

    print("\n--- Scenario 2: Easing Cycle ---")
    easing = simulator.simulate_event(
        event_type="easing",
        quarters=4,
        rate_change_per_quarter=-0.25,
        gdp_growth_assumption=0.005,
    )
    simulator.plot_simulation(easing, title="Easing Cycle: 100bp Cuts")

    print("\n--- Scenario 3: Status Quo ---")
    status_quo = simulator.simulate_event(
        event_type="status_quo",
        quarters=4,
        rate_change_per_quarter=0.0,
        gdp_growth_assumption=0.005,
    )
    simulator.plot_simulation(status_quo, title="Status Quo: No Rate Changes")


if __name__ == "__main__":
    main()
