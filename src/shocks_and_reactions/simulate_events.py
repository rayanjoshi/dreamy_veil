"""Event simulation module for analysing monetary policy shocks on S&P 500 returns.

This module provides a class to:
- Load and preprocess financial data
- Engineer features including shock detection
- Fit an OLS regression model
- Simulate future S&P 500 paths under hypothetical policy shocks
- Visualise simulation results with dual-axis plots

Dependencies:
    - pandas
    - statsmodels.api
    - plotly.graph_objects
    - plotly.subplots
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EventSimulator:
    """A class to simulate S&P 500 reactions to monetary policy shocks.

    This class loads financial data, engineers features (returns, rate changes,
    shock types), fits an OLS regression model, and simulates future price paths
    under hypothetical Federal Reserve rate changes and shock types.

    Attributes
    ----------
        data: The preprocessed DataFrame containing market and policy data.
        model: The fitted OLS regression model (after calling fit_model).
        X: The feature matrix used for model fitting.
        y: The target series (S&P 500 daily returns).
    """

    def __init__(self, data: Path):
        """Initialise the simulator by loading the data.

        Arguments
        ---------
            data: Path to the CSV file containing combined FRED and Yahoo Finance data.
        """
        self.data = pd.read_csv(data, parse_dates=["Date"], index_col="Date")

    def feature_engineering(self) -> None:
        """Perform feature engineering on the loaded data.

        This includes:
        - Calculating daily S&P 500 returns
        - Computing rate changes in basis points
        - Adding lagged returns and M1 money supply growth
        - Classifying FOMC announcement days and shock types

        Returns
        -------
            None
        """
        if "Close" in self.data.columns:
            self.data["SP500_Return"] = self.data["Close"].pct_change()
        if "Effective Federal Funds Rate" in self.data.columns:
            self.data["Rate_Change"] = self.data["Effective Federal Funds Rate"].diff()
        if "Rate_Change" in self.data.columns:
            self.data["Rate_Change_bp"] = self.data["Rate_Change"] * 100

        if "SP500_Return" in self.data.columns:
            self.data["Lagged_Return"] = self.data["SP500_Return"].shift(1)

        if "M1 Money Supply" in self.data.columns:
            self.data["M1_Growth"] = self.data["M1 Money Supply"].pct_change()
            self.data["M1_Growth"] = self.data["M1_Growth"].ffill()
            self.data["Lagged_M1_Growth"] = self.data["M1_Growth"].shift(1)
        else:
            print("Warning: M1 column missing → using 0 as fallback")
            self.data["Lagged_M1_Growth"] = 0.0

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
        )

        self.data["is_fomc_date"] = self.data.index.isin(fomc_dates)

        self.data["Shock_Type"] = "No_Shock"
        mask = self.data["Rate_Change_bp"].abs() >= 10
        self.data.loc[mask & (self.data["Rate_Change_bp"] > 0), "Shock_Type"] = "Hike"
        self.data.loc[mask & (self.data["Rate_Change_bp"] < 0), "Shock_Type"] = "Cut"

        self.data.dropna(inplace=True)

    def fit_model(self):
        """Fit an OLS regression model to predict daily S&P 500 returns.

        Uses rate changes, lagged M1 growth, lagged returns, and shock type dummies
        as predictors. Prints the model summary and stores the fitted model.
        """
        self.model_data = self.data.copy()
        self.model_data = pd.get_dummies(
            self.model_data, columns=["Shock_Type"], drop_first=True, dtype=float
        )

        base_features = ["Rate_Change_bp", "Lagged_M1_Growth", "Lagged_Return"]
        shock_dummies = [
            col for col in self.model_data.columns if col.startswith("Shock_Type")
        ]

        X = self.model_data[base_features + shock_dummies]
        X = sm.add_constant(X)

        y = self.model_data["SP500_Return"]

        model = sm.OLS(y, X).fit()
        print(model.summary())

        self.model = model
        self.X = X
        self.y = y

    def simulate_event(
        self,
        days_ahead: int = 10,
        announcement_rate_change_bp: float = 0.0,
        shock_type: str = "No_Shock",
    ) -> pd.DataFrame:
        """Simulate future S&P 500 paths starting from the most recent data point.

        Simulates daily returns under a hypothetical rate change and shock type on
        the announcement day (day 0). Subsequent days use predicted returns and
        mean reversion dynamics.

        Arguments
        ---------
            days_ahead: Number of days to simulate (default: 10).
            announcement_rate_change_bp: Rate change in basis points on announcement day.
            shock_type: Type of shock ('No_Shock', 'Hike', 'Cut').

        Returns
        -------
            A DataFrame containing simulated dates, returns, and S&P 500 levels.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_model() first.")

        # Extract starting values from the real last date
        last_row = self.data.iloc[-1]
        start_date = last_row.name
        start_sp500 = (
            last_row["Adj Close"] if "Adj Close" in last_row else last_row["Close"]
        )
        start_lagged_return = last_row["Lagged_Return"]
        start_lagged_m1 = last_row["Lagged_M1_Growth"]

        print(f"Simulation starting from last available date: {start_date}")
        print(f"Starting S&P 500 level: {start_sp500:,.2f}")
        print(f"Starting lagged return: {start_lagged_return:.6f}")
        print(f"Starting lagged M1 growth: {start_lagged_m1:.6f}")

        # Prepare simulation
        sim_rows = []
        current_sp500 = start_sp500
        current_lagged_return = start_lagged_return
        current_lagged_m1 = start_lagged_m1

        for day in range(days_ahead):
            # Rate change only applies on the announcement day (day 0)
            rate_this_day = announcement_rate_change_bp if day == 0 else 0.0

            predictions_dict = {
                "const": 1.0,
                "Rate_Change_bp": rate_this_day,
                "Lagged_M1_Growth": current_lagged_m1,
                "Lagged_Return": current_lagged_return,
            }

            # Add shock type dummies (1 for chosen type, 0 otherwise)
            for col in self.X.columns:
                if col.startswith("Shock_Type_"):
                    dummy_name = col.replace("Shock_Type_", "")
                    predictions_dict[col] = (
                        1.0 if (day == 0 and dummy_name == shock_type) else 0.0
                    )

            # Create prediction DataFrame with correct column order
            X_predictions = pd.DataFrame([predictions_dict])[self.X.columns]

            # Predict the return for this day
            predicted_return = self.model.predict(X_predictions)[0]

            # Update current level
            new_sp500 = current_sp500 * (1 + predicted_return)

            # Store results
            sim_rows.append(
                {
                    "Simulated_Date": start_date + pd.Timedelta(days=day + 1),
                    "Day": day,  # 0 = announcement day
                    "Predicted_Daily_Return_%": predicted_return * 100,
                    "Cumulative_Return_%": (new_sp500 / start_sp500 - 1) * 100,
                    "Simulated_SP500_Level": new_sp500,
                }
            )

            current_sp500 = new_sp500
            current_lagged_return = predicted_return

        simulation_df = pd.DataFrame(sim_rows)
        simulation_df = simulation_df.round(4)

        return simulation_df

    def plot_simulation(
        self, simulation_df: pd.DataFrame, title: str = "S&P 500 Simulation"
    ):
        """Plot the simulated S&P 500 levels over the simulation period."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary: Cumulative Return (%)
        fig.add_trace(
            go.Scatter(
                x=simulation_df["Simulated_Date"],
                y=simulation_df["Cumulative_Return_%"],
                mode="lines+markers",
                name="Cumulative Return (%)",
                line=dict(color="blue", width=2),
            ),
            secondary_y=False,
        )

        # Secondary: Daily Predicted Return (%)
        fig.add_trace(
            go.Scatter(
                x=simulation_df["Simulated_Date"],
                y=simulation_df["Predicted_Daily_Return_%"],
                mode="lines+markers",
                name="Daily Predicted Return (%)",
                line=dict(color="orange", dash="dot"),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            yaxis2_title="Daily Return (%)",
            hovermode="x unified",
            legend=dict(yanchor="bottom", xanchor="right"),
            template="plotly_white",
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)

        fig.show()


def main():
    repo_root = Path(__file__).parents[2].resolve()

    simulator = EventSimulator(
        data=Path(repo_root / "data/shocks_and_reactions/combined_data.csv")
    )
    simulator.feature_engineering()
    simulator.fit_model()
    print("\n=== Simulating a 25bp Rate Hike ===")
    hike = simulator.simulate_event(
        days_ahead=10, announcement_rate_change_bp=25.0, shock_type="Hike"
    )
    print(hike)
    simulator.plot_simulation(hike, title="Simulation: 25bp Rate Hike")

    print("\n=== Simulating a 25bp Rate Cut ===")
    cut = simulator.simulate_event(
        days_ahead=10, announcement_rate_change_bp=-25.0, shock_type="Cut"
    )
    print(cut)
    simulator.plot_simulation(cut, title="Simulation: 25bp Rate Cut")

    print("\n=== Simulating No Shock ===")
    base = simulator.simulate_event(
        days_ahead=10, announcement_rate_change_bp=0.0, shock_type="No_Shock"
    )
    print(base)
    simulator.plot_simulation(base, title="Simulation: No Shock")


if __name__ == "__main__":
    main()
