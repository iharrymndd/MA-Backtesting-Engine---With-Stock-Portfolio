# 📈 MA Backtesting Engine — With Stock Portfolio

A Python-based Moving Average (MA) Crossover Backtesting Engine built with **Streamlit**, designed to research, evaluate, and visualize trading strategies across different stocks using historical market data.

The application provides an interactive dashboard where users can explore strategy performance, analyze portfolio results, and evaluate historical trading behavior.

---

## 🚀 Project Overview

The MA Backtesting Engine helps users investigate how Moving Average crossover strategies would have performed on historical stock market data.

It uses Yahoo Finance data through `yfinance` and provides an interactive environment for testing and analyzing trading strategies.

The objective is to make quantitative strategy research more accessible through a customizable dashboard.

## ✨ Key Features

- **Stock Market Data:** Fetch historical market data using Yahoo Finance.
- **Moving Average Strategies:** Research EMA and SMA crossover strategies.
- **Interactive Dashboard:** Explore backtesting results through a Streamlit interface.
- **Performance Analysis:** Evaluate strategy returns and other performance metrics.
- **Portfolio Analysis:** Analyze portfolio-level backtesting results.
- **Visualization:** Interactive charts and performance comparisons using Plotly.
- **Customizable Parameters:** Adjust available strategy and backtesting settings.
- **Export Support:** Export supported results for further analysis.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| yfinance | Historical market data |
| Plotly | Interactive visualizations |
| OpenPyXL | Excel file support |

## 📂 Project Structure

```text
MA-Backtesting-Engine/
│
├── ma_backtesting_engine.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/iharrymndd/MA-Backtesting-Engine---With-Stock-Portfolio.git
```

### 2. Navigate to the project directory

```bash
cd MA-Backtesting-Engine---With-Stock-Portfolio
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run ma_backtesting_engine.py
```

The dashboard should open in your default browser.

## 📊 Backtesting Workflow

1. Select a stock ticker supported by Yahoo Finance.
2. Configure the available backtesting parameters.
3. Fetch historical market data.
4. Run the moving average crossover backtest.
5. Analyze the resulting performance metrics and visualizations.

## ⚠️ Important Notes

- Historical performance does not guarantee future results.
- Yahoo Finance data availability and accuracy may vary.
- Backtesting results depend on the selected parameters and assumptions.
- Transaction costs, slippage, and brokerage assumptions can affect simulated performance.
- This project is intended for research and educational purposes, not as financial advice.

## 🔮 Future Improvements

Potential areas of further development:

- Additional technical indicators.
- More advanced portfolio analytics.
- Enhanced risk and drawdown analysis.
- Additional strategy research tools.
- Improved performance reporting.

## 👨‍💻 Author

**Harish**

GitHub: [@iharrymndd](https://github.com/iharrymndd)

---

*Built as a project to explore quantitative trading, moving average strategies, and systematic backtesting using Python.*