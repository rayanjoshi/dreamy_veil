# dreamy_veil

A Framework for Analysing Monetary Policy Shocks and Financial Market Impacts

<p align="center">
  <img src="https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge" alt="license" />
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="python" />
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="jupyter" />
</p>

<p align="left">Tools and technologies utilised in this project:</p>
<p align="center">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/statsmodels-4051B5?style=for-the-badge" alt="Statsmodels"/>
</p>

**Status (February 2026)**: Projects 1 (equity shocks), 2 (bond/yield curve impacts), and 3 (corporate investment decisions) are complete and fully runnable. **Project 4 (ESG/sustainable finance)** is under active development and coming soon.

This repository provides modular Python tools to study monetary policy shocks and their transmission to financial markets (equities, bonds, and corporate investment), using public data from FRED and Yahoo Finance. It is licensed under the MIT License.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation Notes](#installation-notes)
- [Key Features](#key-features)
- [Workflow Overview](#workflow-overview)
- [License & Data Sources](#license--data-sources)

## Overview

This project explores how monetary policy changes (rate adjustments, FOMC announcements) affect equity markets, bond markets, and corporate investment decisions. It currently implements three completed modules:

1. **Equity reactions** — S&P 500 responses to policy shocks via event studies and simple regressions.
2. **Bond & yield curve reactions** — US Aggregate (AGG) and UK Gilts (IGLT.L) ETF returns linked to policy rates and spreads (US 10Y-2Y, UK 10Y-3M proxy).
3. **Corporate investment decisions** — Panel regression analysis of how Federal Reserve rate changes impact capital expenditure (capex) growth for the Magnificent Seven tech companies, with forward-looking scenario simulations.

Future modules will extend to ESG/sustainability channels (green finance, differential impacts on ESG firms).

The framework emphasises reproducibility, interactive Plotly visualisations, and easy scenario simulation for "what-if" policy experiments.

## Quick Start

```bash
1. # Clone the repository and change into the project directory:

    git clone https://github.com/rayanjoshi/dreamy_veil.git
    cd dreamy_veil

2. # Install dependencies using uv:

    pip install uv
    # or on macOS: brew install uv
    uv sync

3. # Configure FRED credentials in a `.env` file at the repository root:

    # Create .env file with your credentials
    touch .env
    echo "FRED_API_KEY=your_fred_api_key" >> .env

4. # Run the complete workflow via Jupyter notebooks:

    # The notebooks execute the full pipeline: data extracting → analysis → integrated scenarios.
```

## Project Structure

```markdown
dreamy_veil/
├── LICENSE
├── README.md
├── pyproject.toml
├── data/                           # Generated data artifacts
│   ├── corporate_decisions/
│   ├── policy_impacts/
│   └── shocks_and_reactions/
├── notebooks/                      # Primary workflow interface
│   ├── 01_monetary_policy_shocks.ipynb
│   ├── 02_monetary_policy_impacts.ipynb
│   └── 03_corporate_decisions.ipynb
├── src/                           # Core analysis modules
│   ├── corporate_decisions/
│   │   ├── data_analysis.py
│   │   ├── data_loader.py
│   │   └── simulate_events.py
│   ├── policy_impacts/
│   │   ├── data_analysis.py
│   │   ├── data_loader.py
│   │   └── simulate_events.py
│   └── shocks_and_reactions/
│       ├── data_loader.py
│       ├── shock_events.py
│       └── simulate_events.py
```

### Module Integration Workflow

The project is designed for progressive analysis:

1. **Module 1 (Shocks)**: Run first to identify policy shock dates and magnitudes around FOMC announcements (hike/cut/no-shock classification).

2. **Module 2 (Impacts)**: Analyses bond market reactions (US AGG and UK IGLT.L ETF returns) to policy rate changes and yield spread movements using OLS regressions; includes historical visualisations and forward-looking monthly simulations of hypothetical rate scenarios.

3. **Module 3 (Corporate)**: Examines how monetary policy affects corporate capital expenditure using quarterly panel data for the Magnificent Seven tech companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA); features panel regression with ticker fixed effects; includes forward-looking quarterly simulations under tightening, easing, and status quo scenarios.

4. **Module 4 (Sustainable)**: (Upcoming) Will overlay ESG filters and green finance data on prior analyses.

### Important Notes

- **FRED API Key**: Required for monetary policy and macroeconomic data access
- **Data Availability**: Some series may have limited historical coverage; the framework handles missing data via forward-fill and interpolation

## Installation Notes

- This project uses **uv** as the package manager for fast dependency resolution
- Core dependencies include:
  - **pandas/numpy**: Data manipulation and analysis
  - **statsmodels**: Econometric modelling and statistical testing
  - **plotly**: Visualisation and interactive plots
  - **fredapi**: FRED economic data interface
  - **yfinance**: Supplementary market data (Yahoo Finance)

## Key Features

1. **Event-Based Monetary Policy Analysis**  
   Identifies policy shocks via daily rate changes around FOMC announcements and classifies them as hikes, cuts, or no change.

2. **Equity Market Reactions**  
   Estimates S&P 500 daily returns response to policy shocks using OLS with lagged controls; visualises average cumulative returns around events.

3. **Bond Market & Yield Curve Reactions**  
   Analyses US (AGG) and UK (IGLT.L) bond ETF returns in response to policy rate changes and yield spread movements (US 10Y-2Y, UK 10Y-3M proxy); includes historical spread plots and cumulative return comparisons.

4. **Counterfactual Scenario Simulation**  
   Tools to forecast short-term equity paths (daily) and medium-term bond paths (monthly) under hypothetical rate changes (hikes, cuts, QE-style easing, holds), using fitted regression coefficients.

5. **Corporate Investment Transmission Channel**  
   Panel regression analysis of capital expenditure growth responses to Fed rate changes; quarterly forward simulations under tightening/easing scenarios; firm-level heterogeneity analysis.

6. **Modular & Reproducible Structure**  
   Clean separation of data loading, analysis, and simulation modules; interactive Plotly visualisations; easy to extend to new assets or policy variables.

7. **Extensible Framework**  
   Designed for future expansion to ESG/sustainability channels (Project 4).

## Workflow Overview

### Current Workflow (Projects 1, 2, & 3)

1. **Data Collection** — FRED macro/policy series + yfinance market data + company fundamentals
2. **Shock Identification** — Classify rate changes around FOMC dates (Project 1)
3. **Market Impact Analysis** — OLS regressions + historical plots (equity & bonds, Projects 1 & 2)
4. **Corporate Impact Analysis** — Panel regressions on capex growth with ticker fixed effects (Project 3)
5. **Scenario Simulation** — Forecast paths under hypothetical policy scenarios:
   - Daily equity paths (Project 1)
   - Monthly bond paths (Project 2)  
   - Quarterly corporate capex paths (Project 3)

### Planned Extensions

- ESG/green finance overlays (Project 4)

## License & Data Sources

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project uses data from multiple sources:

- **FRED Economic Data**: Public domain (Federal Reserve Bank of St. Louis)

- **Yahoo Finance**: Subject to Yahoo Finance API terms
