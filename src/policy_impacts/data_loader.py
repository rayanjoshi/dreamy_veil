"""Module for loading and processing economic and market data related to policy impacts.

This module provides a ``DataLoader`` class that fetches historical data from the Federal Reserve Economic Data
(FRED) service and Yahoo Finance, aligns the datasets on trading days, computes derived variables such as the UK
Bond Yield Spread, and saves the cleaned combined dataset to a CSV file in the project's data directory.

"""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from fredapi import Fred
import yfinance as yf

load_dotenv()

pd.set_option('future.no_silent_downcasting', True)


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
        self.data_dir = repo_root / "data" / "policy_impacts"
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

        #combined_data.dropna(inplace=True)

        self._save_data(combined_data)

    def _load_fred_data(self) -> pd.DataFrame:
        """Load economic indicators from FRED.

        Retrieves the 10-Year Minus 2-Year Treasury Yield Spread (T10Y2Y) and the
        UK Bond Yield Spread (IR3TIB01GBM156N - IRLTLT01GBM156N)
        for the period 2020-01-01 to 2025-12-31.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the two FRED series, combined and forward-filled.
        """
        fred_api_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_api_key)

        ten_year_minus_two_year = fred.get_series(
            series_id="T10Y2Y",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )
        interbank_rates = fred.get_series(
            series_id="IR3TIB01GBM156N",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )
        
        uk_10_year_rates = fred.get_series(
            series_id="IRLTLT01GBM156N",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )
        
        fed_funds = fred.get_series(
            series_id="FEDFUNDS",
            observation_start="2020-01-01",
            observation_end="2025-12-31",
        )
        

        ten_year_minus_two_year = ten_year_minus_two_year.to_frame(name="10Y-2Y Treasury Yield Spread")
        interbank_rates = interbank_rates.to_frame(name="IR3TIB01GBM156N")
        uk_10_year_rates = uk_10_year_rates.to_frame(name="UK 10-Year Government Bond Yield")
        fed_funds = fed_funds.to_frame(name="Federal Funds Rate")
        uk_spread = self._combine_data(interbank_rates, uk_10_year_rates)
        uk_spread['UK Bond Yield Spread'] = uk_spread['UK 10-Year Government Bond Yield'] - uk_spread['IR3TIB01GBM156N']

        fred_data = self._combine_data(ten_year_minus_two_year, uk_spread)
        fred_data = self._combine_data(fred_data, fed_funds)

        return fred_data

    def _load_yf_data(self) -> pd.DataFrame:
        """Load S&P 500 historical data from Yahoo Finance.

        Downloads daily data for the AGG and IGLT.L tickers from 2020-01-01 to 2025-12-31.

        Returns
        -------
        pd.DataFrame
            The downloaded Yahoo Finance data.

        Raises
        ------
        ValueError
            If no data is fetched or the DataFrame is empty.
        """
        ticker_agg = "AGG"
        ticker_iglt = "IGLT.L"

        yf_data_agg = yf.download(
            tickers=ticker_agg,
            start="2020-01-01",
            end="2025-12-31",
            progress=False,
            multi_level_index=False,
        )

        if yf_data_agg is None or yf_data_agg.empty:
            raise ValueError(f"No data fetched from Yahoo Finance for ticker {ticker_agg}")
        
        yf_data_iglt = yf.download(
            tickers=ticker_iglt,
            start="2020-01-01",
            end="2025-12-31",
            progress=False,
            multi_level_index=False,
        )

        if yf_data_iglt is None or yf_data_iglt.empty:
            raise ValueError(f"No data fetched from Yahoo Finance for ticker {ticker_iglt}")

        # Clean ticker strings for safe column suffixes
        def _clean_ticker(s):
            return s.replace('.', '_').replace('-', '_')

        # Append ticker-specific suffix to each column to avoid overlapping names
        yf_data_agg.columns = [f"{col}_{_clean_ticker(ticker_agg)}" for col in yf_data_agg.columns]
        yf_data_iglt.columns = [f"{col}_{_clean_ticker(ticker_iglt)}" for col in yf_data_iglt.columns]

        yf_data = self._combine_data(yf_data_agg, yf_data_iglt)

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

        data = data.ffill().infer_objects(copy=False)

        data.dropna(how="all", inplace=True)
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