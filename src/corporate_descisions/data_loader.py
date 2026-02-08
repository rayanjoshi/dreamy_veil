"""Module for downloading and combining financial data for analysis.

This module fetches stock price data for the 'Magnificent Seven' companies,
macroeconomic indicators from FRED (Federal Reserve Economic Data),
and selected fundamental metrics, then combines them into a single
time-series DataFrame saved as a CSV file.

Data sources:
    - Yahoo Finance (via yfinance)
    - FRED (Federal Reserve Economic Data via fredapi)
"""
import os
from pathlib import Path
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()


class DataLoader:
    """Class responsible for fetching and combining financial and macroeconomic data.

    This class handles the creation of a data directory and the retrieval
    of stock prices, Federal Funds Rate, real GDP, and selected fundamental
    metrics for the Magnificent Seven companies. The resulting dataset is
    saved as a CSV file in the project's data directory.
    """
    def __init__(self) -> None:
        """Initialise the DataLoader and create the data storage directory.
        """
        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = repo_root / "data" / "corporate_decisions"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Fetch financial and macroeconomic data and save to CSV.

        This method:
            - Downloads daily closing prices for the Magnificent Seven stocks
            - Retrieves Federal Funds Rate and real GDP from FRED
            - Collects total assets, liabilities and capital expenditure data
            - Calculates debt-to-assets ratio and GDP growth
            - Resamples all series to business-day frequency with forward-fill
            - Combines all data into a single DataFrame
            - Saves the result as 'combined_data.csv'

        Returns
        -------
        None
            The combined dataset is saved to disk; no value is returned.
        """
        magnificent_seven = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        start_date = "2020-01-01"
        end_date = "2025-12-31"

        fred_api_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_api_key)

        fed_funds = fred.get_series(
            "FEDFUNDS", observation_start=start_date, observation_end=end_date
        )
        gdp = fred.get_series(
            "GDPC1", observation_start=start_date, observation_end=end_date
        )

        fed_funds_q = fed_funds.resample("B").last().ffill()
        gdp_q = gdp.resample("B").last().ffill()

        fred_data = pd.DataFrame(
            {
                "FedFunds": fed_funds_q,
                "GDP": gdp_q,
                "Delta_FedFunds": fed_funds_q.diff(),
                "GDP_Growth": gdp_q.pct_change(),
            }
        )

        yf_data = yf.download(
            magnificent_seven, start=start_date, end=end_date, progress=False
        )

        if yf_data is None or yf_data.empty:
            print("No data fetched from Yahoo Finance.")
            return

        if isinstance(yf_data.columns, pd.MultiIndex):
            stock_prices_q = yf_data["Close"].resample("B").last()
        else:
            stock_prices_q = yf_data[["Close"]].resample("B").last()

        fundamentals_list = []

        for symbol in magnificent_seven:
            ticker = yf.Ticker(symbol)

            try:
                cashflow = ticker.cashflow
                balance = ticker.balance_sheet
            except Exception as e:
                print(f"    Error fetching data for {symbol}: {e}")
                continue

            if cashflow.empty or balance.empty:
                print(f"    No fundamental data for {symbol}")
                continue

            capex_data = cashflow.loc["Capital Expenditure"]

            fund_df = pd.DataFrame(
                {
                    f"{symbol}_total_assets": balance.loc["Total Assets"],
                    f"{symbol}_total_liabilities": balance.loc[
                        "Total Liabilities Net Minority Interest"
                    ],
                }
            )

            if capex_data is not None:
                fund_df[f"{symbol}_capex"] = capex_data

            fund_df[f"{symbol}_debt_to_assets"] = (
                fund_df[f"{symbol}_total_liabilities"]
                / fund_df[f"{symbol}_total_assets"]
            )

            fund_df.index = pd.to_datetime(fund_df.index)
            fund_df = fund_df.sort_index()
            fund_df_q = fund_df.resample("B").last().ffill(limit=8)
            fundamentals_list.append(fund_df_q)

        combined_data = fred_data.copy()

        for symbol in magnificent_seven:
            if symbol in stock_prices_q.columns:
                combined_data[f"{symbol}_close"] = stock_prices_q[symbol]

        for fund_df in fundamentals_list:
            combined_data = combined_data.join(fund_df, how="outer")

        combined_data = combined_data.sort_index()
        combined_data = combined_data.loc[start_date:end_date]
        combined_data.to_csv(self.data_dir / "combined_data.csv")


def main() -> None:
    """Entry point for the data loading script.

    Creates a DataLoader instance and triggers the data collection process.
    """
    data_loader = DataLoader()
    data_loader.load_data()


if __name__ == "__main__":
    main()
