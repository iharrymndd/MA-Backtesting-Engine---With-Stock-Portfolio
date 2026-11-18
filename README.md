# 📈 MA Backtesting Engine — With Stock Portfolio

A Python/Streamlit dashboard for researching **EMA / SMA crossover strategies** on daily Yahoo Finance prices, for a single stock or a multi-stock portfolio. You choose the parameters, run the backtest, and get the equity curve, drawdown, trade log and performance metrics.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Interactive dashboard |
| pandas / NumPy | Data and numerical computation |
| yfinance | Historical market data |
| Plotly | Interactive charts and heat maps |
| numba (optional) | JIT-compiled position logic for fast sweeps |
| OpenPyXL | Excel export |

---

## 📂 Project Structure

```text
MA-Backtesting-Engine---With-Stock-Portfolio/
│
├── ma_backtesting_engine.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/iharrymndd/MA-Backtesting-Engine---With-Stock-Portfolio.git
cd MA-Backtesting-Engine---With-Stock-Portfolio
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
pip install numba            # optional, speeds up sweeps
streamlit run ma_backtesting_engine.py
```

Nothing is calculated until a Run button is pressed, so changing widgets only redraws the page.

---

## 🧭 Project Walkthrough

### 1. Data
Enter tickers in the sidebar (for example `RELIANCE, TCS, INFY`) and pick a market suffix (NSE `.NS`, BSE `.BO`, or as typed). The app downloads each ticker's auto-adjusted daily prices once and caches them. It also downloads extra history before each start date so that every indicator, up to a 400-bar SMA, is fully warmed up on day one.

### 2. The strategy
- **Signal:** a long entry when the EMA crosses above the SMA, and a short entry when it crosses below. Shorts can be switched off.
- **Exit:** on the opposite crossover, optionally confirmed by the close being beyond the EMA. A pending exit is cancelled if the EMA moves back to its original side.
- **Timing:** the signal is computed at the close of bar *t*, and the position earns returns from bar *t+1*, so no bar uses its own future return.
- **Costs and risk:** a per-side trading cost, charged twice on a flip, and an optional stop loss that exits at the first close beyond the stop level.
- **Trading timeline:** trading can be restricted to chosen date ranges. Outside them the strategy stays flat.

### 3. No-trade filters
Three optional filters detect sideways markets:
- **Efficiency Ratio:** no trade when the ratio falls below a threshold.
- **ADX:** no trade when the ADX falls below a threshold.
- **Vortex:** no trade when +VI is below −VI, using a 14-bar period.

On a flagged bar, any open trade is closed and no new entry is taken. The charts shade these zones.

### 4. Section 1: estimating parameters
Instead of guessing the EMA and SMA lengths, they can be estimated on a training window.
- The engine backtests every EMA < SMA pair in a grid and draws three heat maps: annual return, max drawdown, and return/drawdown.
- It then picks a pair by one of five methods: highest raw return, best neighbourhood average (a stable plateau), mean-minus-std of neighbours, return/drawdown averaged over neighbours, or most consistent year-to-year return.
- The plateau methods avoid picking a single lucky cell that wouldn't repeat out of sample.

### 5. Section 2: the backtest
The chosen pair, or manual lengths, is run on a separate backtest window. The page then shows:
- the price chart with EMA, SMA, entries, exits and the filter indicators below it;
- the strategy equity curve, gross and net of costs, against buy-and-hold;
- a drawdown chart and a full trade log, where every trade carries its exit reason (crossover, stop loss, or no-trade zone);
- metrics: net PnL, total and annualised return, max drawdown, Sharpe, Sortino, Calmar, volatility, win rate, profit factor, expectancy, payoff ratio, exposure, trading cost, and crossover statistics.

### 6. Permutations
A parameter-sweep section sits at the top of the page. For each of 13 parameters you choose Fixed, Range, Specific values or All. The engine backtests every valid combination and returns one row per combination.
- Results can be sorted by any of 27 metrics, limited to the top N, and ticked for export.
- A two-parameter pivot heat map shows how two settings interact.
- The position logic is JIT-compiled with numba, when installed, and each moving average is computed once per length, so large sweeps stay fast.
- A best-fit row can be copied into a ticker's profile.

### 7. Portfolio mode
With several tickers, each stock gets its own parameter profile and trades independently on its own share of the capital.
- **Capital split:** equal, or by inverse volatility. Inverse-volatility weights use only data from before the backtest start and stay fixed, with no rebalancing.
- **Portfolio results:** return, volatility, Sharpe, Sortino, drawdown and Calmar come from the combined equity curve, so diversification is already reflected.
- **Analytics:** diversification ratio, pairwise correlation of the strategy returns, each stock's contribution to return, and its share of portfolio risk. Each stock's growth is also charted.
- **Switching tickers:** a ticker switcher shows each stock's chart, parameters and own results.

### 8. Export
Parameters, metrics, per-stock tables, correlations and selected permutation runs download as one Excel workbook or as CSV.

### 9. Design notes
- Results are kept in session state, and the app warns when the settings have changed since the last run.
- The code uses frozen dataclasses and type hints, and falls back to plain Python when numba isn't installed.

---

## 📌 Example Results

> *Add one finished run here: tickers, training and backtest dates, the pair chosen by section 1, and out-of-sample annualised return vs buy-and-hold, max drawdown and Sharpe, with a dashboard screenshot.*

---

## ⚠️ Assumptions & Limitations

- Entries and exits are filled at the signal bar's close.
- Prices are Yahoo Finance auto-adjusted daily data, whose accuracy can vary, and tickers are today's constituents.
- Costs are a flat percentage per side.
- Short positions are allowed in the simulation on any stock.
- Validation is a single train/test split. If the training and backtest dates overlap, the app warns that the result is not fully out of sample.
- Portfolio weights are fixed at the start.
- Permutations split capital equally and default to the backtest dates.
- Annualised figures are approximations, and historical performance does not guarantee future results.
- For research and education only, not financial advice.

---

## 👨‍💻 Author

**Harish Mundada** — GitHub: [@iharrymndd](https://github.com/iharrymndd)
