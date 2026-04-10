# market-intelligence-tool
A Python-based financial analysis tool for multi-asset portfolio analysis, 
risk measurement, and portfolio optimization.

Built as a personal project to apply quantitative finance concepts 
from J. Hull (*Options, Futures & Other Derivatives*) and J. Murphy 
(*Technical Analysis of the Financial Markets*).

---

## Features

### Asset Analysis
- Daily and annualized return statistics
- Volatility (daily and annualized)
- Skewness and Kurtosis (distribution shape analysis)
- Moving averages (MA20, MA50) with Death Cross / Golden Cross detection

### Risk Metrics
- **VaR** (Value at Risk) — historical method, 95% and 99% confidence levels
- **CVaR / Expected Shortfall** — average loss beyond VaR threshold (Basel III)
- **Sharpe Ratio** — risk-adjusted return
- **Max Drawdown** — maximum peak-to-trough loss

### Portfolio Analysis
- Multi-asset support (N assets)
- Equally-weighted portfolio metrics
- **Markowitz optimization** — matrix algebra (w^T · Σ · w)
- **Optimal Sharpe portfolio** — scipy optimization
- **Monte-Carlo simulation** — 5,000 random portfolios
- Correlation heatmap

### Backtesting
- Rolling VaR backtesting (60-day window)
- Exception counting vs Basel III threshold (max 12 exceptions / 252 days)

---

## Financial Concepts

| Concept | Description |
|---|---|
| VaR 95% | Maximum loss in 95% of cases |
| CVaR | Average loss in the worst 5% of cases — Basel III metric |
| Sharpe Ratio | (Rp - Rf) / σp — return per unit of risk |
| Max Drawdown | (Trough - Peak) / Peak |
| Markowitz | σ²p = w^T · Σ · w — portfolio variance via covariance matrix |
| Monte-Carlo | 5,000 random portfolios to map the efficient frontier |
| VaR Backtesting | Regulatory validation of VaR model (Basel III) |

---

## Technologies

- Python 3.x
- pandas
- numpy
- scipy
- matplotlib
- yfinance

---

## Installation

```bash
git clone https://github.com/ugodelpuppo/market-intelligence-tool.git
cd market-intelligence-tool
pip install -r requirements.txt
cd src
python main.py
```

---

## Usage

When you run the tool, it will prompt you for:

The tool will generate:
- Individual statistics and risk metrics for each asset
- 4 charts per asset (prices + MA, return distribution, drawdown, VaR backtest)
- Portfolio-level analysis (correlation heatmap, Monte-Carlo, optimal Sharpe)

---

## Project Structure
market-intelligence-tool/
│
├── src/
│   ├── main.py            # Entry point
│   ├── data_loader.py     # Price data loading (yfinance)
│   ├── indicators.py      # Financial metrics and risk measures
│   ├── asset_profile.py   # Asset information
│   └── plotter.py         # Visualizations
│
├── requirements.txt
└── README.md

---

## Limitations & Future Improvements

- VaR historical method assumes past distribution represents future risk
- Volatility clustering violates i.i.d. assumption → GARCH model would improve calibration
- Optimal weights are sensitive to historical period (Markowitz instability)
- Data limited to Yahoo Finance availability

---

## Author

**Ugo Del Puppo**  
M1 Finance — Grenoble École de Management  
AMF Certified  
[LinkedIn](https://www.linkedin.com/in/ugo-del-puppo)
