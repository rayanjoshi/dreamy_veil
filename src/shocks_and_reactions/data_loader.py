"""Module for loading and processing economic and market data related to monetary shocks and market reactions.

This module provides a ``DataLoader`` class that fetches historical data from the Federal Reserve Economic Data
(FRED) service and Yahoo Finance, aligns the datasets on S&P 500 trading days, computes key derived variables
such as S&P 500 daily returns and changes in the effective federal funds rate, and saves the cleaned combined
dataset to a CSV file in the project's data directory.

The script can be run directly to execute the data loading process.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from fredapi import Fred
import yfinance as yf

load_dotenv()


class DataLoader:
    """A class responsible for loading, combining, and processing economic and market data.

    It fetches data from FRED (Federal Reserve Economic Data) and Yahoo Finance,
    aligns the datasets on trading days, computes returns and changes, and saves
    the combined dataset to a CSV file.
    """

    def __init__(self) -> None:
        """Initialise the DataLoader instance.

        Determines the repository root directory and sets up the data directory
        for saving the combined dataset.
        """
        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = repo_root / "data" / "shocks_and_reactions"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load, process, and save the combined economic and market dataset.

        Fetches data from FRED and Yahoo Finance, aligns them, computes S&P 500
        returns and federal funds rate changes, drops missing values, and saves
        the result to CSV.
        """
        fred_data = self._load_fred_data()
        yf_data = self._load_yf_data()

        trading_days = yf_data.index
        fred_data = fred_data.reindex(trading_days)

        combined_data = self._combine_data(fred_data, yf_data)

        combined_data["SP500_Return"] = combined_data["Close"].pct_change()
        combined_data["Rate_Change"] = combined_data[
            "Effective Federal Funds Rate"
        ].diff()
        combined_data.dropna(inplace=True)

        self._save_data(combined_data)

    def _load_fred_data(self) -> pd.DataFrame:
        """Load economic indicators from FRED.

        Retrieves the Effective Federal Funds Rate (DFF) and M1 Money Supply (M1SL)
        for the period 2020-01-01 to 2025-12-31.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the two FRED series, combined and forward-filled.
        """
        fred_api_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_api_key)

        effective_rate = fred.get_series(
            series_id="DFF",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )
        m1_money_supply = fred.get_series(
            series_id="M1SL",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )

        effective_rate = effective_rate.to_frame(name="Effective Federal Funds Rate")
        m1_money_supply = m1_money_supply.to_frame(name="M1 Money Supply")

        fred_data = self._combine_data(effective_rate, m1_money_supply)

        return fred_data

    def _load_yf_data(self) -> pd.DataFrame:
        """Load S&P 500 historical data from Yahoo Finance.

        Downloads daily data for the ^GSPC ticker from 2020-01-01 to 2025-12-31.

        Returns
        -------
        pd.DataFrame
            The downloaded Yahoo Finance data.

        Raises
        ------
        ValueError
            If no data is fetched or the DataFrame is empty.
        """
        yf_data = yf.download(
            tickers="^GSPC",
            start="2020-01-01",
            end="2025-12-31",
            progress=False,
            multi_level_index=False,
        )

        if yf_data is None or yf_data.empty:
            raise ValueError("No data fetched from Yahoo Finance for ticker ^GSPC")

        return yf_data

    def _combine_data(
        self, dataset_1: pd.DataFrame, dataset_2: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine two DataFrames by joining on their indices.

        Performs an outer join, sorts by date, forward-fills missing values,
        and drops any remaining rows with NaN.

        Parameters
        ----------
        dataset_1 : pd.DataFrame
            The first dataset to combine.
        dataset_2 : pd.DataFrame
            The second dataset to combine.

        Returns
        -------
        pd.DataFrame
            The combined, cleaned DataFrame with 'Date' as the index name.
        """
        data = dataset_1.join(dataset_2, how="outer")
        data.sort_index(inplace=True)
        data = data.ffill()
        data.dropna(inplace=True)
        data.index.name = "Date"
        return data

    def _save_data(self, combined_data: pd.DataFrame) -> None:
        """Save the combined DataFrame to a CSV file.

        Parameters
        ----------
        combined_data : pd.DataFrame
            The processed DataFrame to save.
        """
        combined_data.to_csv(self.data_dir / "combined_data.csv")


def main():
    """Entry point for running the data loading process."""
    data_loader = DataLoader()
    data_loader.load_data()


if __name__ == "__main__":
    main()
