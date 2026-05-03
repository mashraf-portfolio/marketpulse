# MarketPulse — Multi-Task Financial Forecasting

A demonstration of three forecasting heads built on a single shared data pipeline:

- **Tab 1 — Price/Direction:** ARIMA, Prophet, LSTM, and Temporal Fusion Transformer (TFT). Click the TFT model to see attention-based explainability.
- **Tab 2 — Volatility:** GARCH(1,1) and LSTM-vol forecasts with a derived high-vol binary classifier.
- **Tab 3 — Regime:** Bull / bear / sideways classification via HMM and XGBoost.

## Live demo notes

- This Space uses 23 pre-cached tickers across 8 markets (US, Saudi Arabia, Kuwait, Qatar, Egypt, UAE, Netherlands).
- First request after wake from sleep takes 30-60 seconds. Subsequent requests are instant.
- All forecasts ship with confidence intervals. Predictions are educational, not investment advice.

## More

Full source code, walk-forward validation results, and architecture documentation:
**[github.com/mashraf-portfolio/marketpulse](https://github.com/mashraf-portfolio/marketpulse)**
