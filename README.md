Live Demo: https://huggingface.co/spaces/SpencerCPurdy/Quantitative_Alpha_Mining_Platform_with_LLM_Discovery

# Quantitative Alpha Mining Platform with LLM Discovery

This project is a sophisticated simulation platform designed to demonstrate a complete quantitative research and trading workflow. It showcases the discovery, evaluation, and implementation of alpha factors using a combination of classical methods, modern machine learning techniques, and optional Large Language Model (LLM) integration. The entire system is controlled through a comprehensive, interactive user interface.

This platform operates on **synthetically generated market data** and is built for **demonstration and educational purposes only**. It does not connect to live markets or provide financial advice.

## Core Features

* **Multi-Source Alpha Factor Generation**: The platform discovers potential trading signals (alpha factors) from three distinct sources:
    * **Classical Factors**: Implements a suite of quantitative factors inspired by the "101 Formulaic Alphas" paper, focusing on price and volume relationships.
    * **LLM-Generated Factors**: Uses OpenAI's GPT models to generate novel factor formulas based on market context. If an API key is not provided, the system utilizes a robust fallback method to generate structurally similar factors.
    * **Alternative Data Factors**: Simulates text from earnings calls, news, and SEC filings, and then performs sentiment analysis using the `ProsusAI/finbert` model to create sentiment-based factors.

* **Comprehensive Factor Evaluation**: Every discovered factor is rigorously tested using a suite of industry-standard metrics, including:
    * **Information Coefficient (IC)**: Measures the predictive power of a factor.
    * **IC Decay**: Analyzes how quickly a factor's predictive power diminishes over time.
    * **Sharpe Ratio**: Calculates the risk-adjusted return of a factor-based portfolio.
    * **Turnover**: Measures the trading frequency implied by the factor, which is critical for cost analysis.
    * **Maximum Drawdown**: Assesses the largest peak-to-trough decline.

* **Market Regime Detection**: The platform employs statistical methods (such as trend strength and the Hurst exponent) to classify the market into one of four regimes: **Trending Up**, **Trending Down**, **Mean Reverting**, or **Volatile**.

* **Regime-Aware Portfolio Construction**: Portfolio allocation is not static. It dynamically adapts to the current market regime using a sophisticated two-layer optimization process:
    1.  **Hierarchical Risk Parity (HRP)** is used to allocate weights to factors within the same category (e.g., momentum, volatility), minimizing correlation.
    2.  **Regime-based weights** are then applied across categories, overweighting factor types that are expected to perform well in the detected regime.

* **Realistic Portfolio Backtesting**: A full backtest engine simulates portfolio performance over the historical data, accounting for transaction costs and periodic rebalancing.

* **Interactive Dashboard**: The entire platform is controlled via a Gradio-based interface with three distinct tabs for discovery, in-depth analysis, and backtesting.

## How It Works

The platform follows a logical workflow that mirrors the process used in quantitative hedge funds:

1.  **Data Generation**: The `MarketDataGenerator` creates a synthetic OHLCV (Open, High, Low, Close, Volume) time series with embedded, realistic market regimes.
2.  **Factor Discovery**: The `AlphaMiningPlatform` calls on the `ClassicalAlphaFactors`, `LLMAlphaGenerator`, and `AlternativeDataPipeline` modules to create a large universe of potential alpha factors.
3.  **Evaluation and Selection**: The `FactorEvaluator` calculates performance and risk metrics for every factor. The platform then filters this universe down to a smaller set of **active factors** based on user-defined thresholds for predictive power (IC) and correlation to ensure diversification.
4.  **Portfolio Construction**: Using the current market regime detected by the `MarketRegimeDetector`, the `RegimeAwarePortfolioOptimizer` calculates the optimal weight for each active factor using the HRP methodology.
5.  **Backtesting and Analysis**: The user can run a backtest on the final portfolio. The system simulates the strategy's historical performance, providing detailed charts and metrics. Users can also perform deep-dive analyses on individual factors, such as visualizing their IC decay.

## Technical Stack

* **Machine Learning / Quantitative Analysis**: PyTorch, scikit-learn, Transformers, OpenAI
* **Data Manipulation and Computation**: NumPy, Pandas, SciPy, StatsModels
* **Web Interface and Dashboard**: Gradio
* **Visualization**: Plotly, Seaborn
* **Technical Analysis**: ta

## How to Use the Demo

The interface is organized into three tabs:

1.  **Factor Discovery**:
    * Adjust the configuration sliders for market data length, number of factors to generate, and the minimum IC threshold for selection.
    * (Optional) Enter an OpenAI API key to enable true LLM-based factor generation.
    * Click **Generate & Evaluate Factors**.
    * Analyze the resulting heatmap, distribution plots, and summary statistics to understand the factor landscape.

2.  **Factor Analysis**:
    * After running the discovery step, select a factor from the dropdown menu.
    * Click **Analyze Factor Decay** to view a plot showing how that factor's predictive power evolves over time.

3.  **Portfolio Backtest**:
    * Once a portfolio of active factors has been created in the first tab, set your desired initial capital and rebalancing frequency here.
    * Click **Run Backtest**.
    * Review the portfolio's equity curve against a buy-and-hold benchmark and analyze the key performance metrics.

## Disclaimer

This project is a simulation designed for demonstration and is not intended for real-world trading. All market data is synthetically generated. The strategies and analyses presented are for educational purposes to showcase skills in quantitative development and machine learning and should not be considered financial advice. Financial markets involve substantial risk.
