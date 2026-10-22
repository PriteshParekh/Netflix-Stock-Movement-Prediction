# Netflix Stock Movement Prediction

This project explores whether historical Netflix stock behaviour,
app-review sentiment and content-release activity can help predict
Netflix’s next trading-day stock direction.

The work covers data inspection and preparation, feature engineering,
model comparison, hyperparameter tuning and a reusable scikit-learn
pipeline.

## Objective

The main objective is to investigate next-trading-day stock direction
using historical market information, with review and release-related
data included as additional potential predictors.

This is a supervised binary classification task:

- 1: The next trading day’s return is positive.
- 0: The next trading day’s return is zero or negative.

## Dataset

Source: [Netflix 360: Daily Stock, App Sentiment & Releases — Kaggle](https://www.kaggle.com/datasets/samartalwar/netflix-360-daily-stock-app-sentiment-and-releases)

The dataset combines:
- Stock prices, trading volume and daily returns.
- App-review counts, ratings and sentiment.
- Content-release counts and recorded IMDb ratings.
- Dates and trading-day indicators.

## Approach

The project follows a structured workflow:

1. Inspect the dataset’s structure, data types and date coverage.
2. Check missing values, duplicates, variable ranges, trading-day
   records and gaps in the data.
3. Create historical features using lags and rolling windows.
4. Separate observations into chronological training, validation
   and test periods.
5. Compare classification models against a majority-class baseline.
6. Tune the selected approach using development data.
7. Build and verify a reusable preprocessing and prediction pipeline.
8. Evaluate the fixed approach on the held-out test period.

The final model uses 14 engineered features covering historical
returns, volatility, trading volume, review activity, sentiment,
ratings and release activity.

## Final Pipeline

The selected approach is a Gradient Boosting classifier.

The scikit-learn pipeline:
- Selects the approved feature columns.
- Fills missing values using medians learned during fitting.
- Predicts next-trading-day stock direction.

Historical feature creation remains a separate preparation stage.
The pipeline accepts engineered feature tables rather than raw data.

## Results

Final fitting uses 1,628 training and validation observations for
both the selected model and the majority-class baseline.
Evaluation uses 282 held-out test observations.

| Metric | Majority-Class Baseline | Gradient Boosting |
|---|---:|---:|
| Accuracy | 52.48% | 53.55% |
| Balanced accuracy | 50.00% | 53.34% |
| ROC AUC | 0.5000 | 0.5416 |

The model correctly predicts 151 outcomes, compared with 148 for
the baseline. This represents a modest improvement of approximately
1.06 percentage points in accuracy.

The combined model’s comparison with the baseline does not, by itself,
establish whether review, sentiment or release-related features add
value beyond market features alone.

## Findings and Learning

The project demonstrates a complete machine-learning workflow,
from preparing data and engineering features to comparing models
and building a reusable pipeline.

Comparison with a simple baseline and assessment across multiple
metrics provide a clearer view of performance than accuracy alone.

The selected model shows a modest improvement, but the results
do not demonstrate a reliable forecasting advantage.

## Limitations

- Final results cover one held-out test period.
- The evaluation does not establish statistical significance.
- Historical availability of source information requires careful
  consideration when interpreting the results.
- This is a classification experiment, not a trading backtest.
  The evaluation excludes profitability, transaction costs
  and execution risk.

## Running the Project

1. Clone this repository.
2. Obtain the dataset from the source linked above.
3. Check that the notebook’s data-loading path matches your local setup.
4. Open `netflix-stock-movement-prediction.ipynb` in Jupyter.
5. Run the cells in order.

The notebook uses Python, pandas, NumPy, Matplotlib and scikit-learn.

## Conclusion

The project provides a reusable pipeline and a documented evaluation
of next-trading-day Netflix stock-direction prediction.

It offers a foundation for further investigation, including evaluation
across additional market periods and closer assessment of the contribution
of each feature group. It is not intended as investment advice or a
trading-ready model.
