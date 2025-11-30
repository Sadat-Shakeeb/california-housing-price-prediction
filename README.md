# California Housing Price Prediction

A complete end-to-end machine learning project to predict median house values using the scikit-learn California Housing dataset.
The workflow includes preprocessing, feature selection, model comparison, hyperparameter tuning, interpretation, and saving a final deployable model.

---

## Dataset

Dataset loaded using `fetch_california_housing(as_frame=True)`.

It contains 8 numerical features describing California districts, such as:

* `MedInc` – Median income
* `HouseAge` – Median house age
* `AveRooms` – Average rooms per household
* `Latitude`, `Longitude`
* …and others

Target variable:
**`MedHouseVal`** – Median house value.

---

# Project Workflow

## 1. Load and Explore the Dataset

Used `fetch_california_housing(as_frame=True)` and basic pandas methods (`head()`, `info()`, `describe()`) to inspect the data.

---

## 2. Train–Test Split

Split the dataset using:

* `train_test_split(test_size=0.2, random_state=42)`
* 80% training, 20% test
* Ensures reproducibility and unbiased evaluation

---

## 3. Baseline Models

Built three initial baseline models:

* `LinearRegression`
* `Ridge`
* `RandomForestRegressor`

### Baseline Results

| Model             | R²                     | RMSE                   |
| ----------------- | ---------------------- | ---------------------- |
| Linear Regression | 0.5749693819995386     | 0.7463001660559125     |
| Ridge Regression  | 0.5749977945514843     | 0.7462752212099223     |
| Random Forest     | **0.8051230593157366** | **0.5053399773665033** |

**Interpretation:**
Random Forest provides the best performance with a large margin.
Linear and Ridge regressions perform similarly.

---

## 4. Ridge Cross-Validation

Used:

```python
cross_val_score(model, X, y, cv=5, scoring="r2")
```

### CV Scores

```
[0.55722286, 0.46693596, 0.55175024, 0.53689198, 0.52531766]
Mean CV R²: 0.5276237392612451
```

Indicates moderate variance across folds and slight underfitting.

---

## 5. Full Pipeline (Preprocessing + Modeling)

Constructed a full preprocessing + Ridge model pipeline:

* `SimpleImputer(strategy="median")`
* `StandardScaler()`
* `SelectKBest(f_regression)`
* `Ridge()`

This prevents data leakage and ensures clean reproducibility.

---

## 6. Hyperparameter Tuning with GridSearchCV

Grid search space:

```
select__k: [4, 6, 8]
model__alpha: [0.1, 1.0, 10.0]
```

Performed:

* 5-fold cross-validation
* Scoring: R²
* Total models trained: 9 × 5 = 45 fits

### Best Parameters

```
{'model__alpha': 0.1, 'select__k': 8}
```

### Best Mean CV R²

```
0.6114839657407327
```

Hyperparameter tuning significantly improved CV performance.

---

## 7. Final Test Performance (Best Ridge Model)

After training the best estimator on the full training set:

```
R² = 0.5757905180002312
RMSE = 0.7455789118982769
```

Performance is similar to baseline Ridge but with better regularization and feature selection control.

---

## 8. Coefficient Interpretation

A simplified Ridge model was built (without feature selection) using the best alpha:

* Imputation → Scaling → Ridge(alpha = 0.1)
* Coefficients extracted using `model.coef_`

This allows identifying which features positively or negatively impact predicted house values.

Positive coefficient → increases predicted price
Negative coefficient → decreases predicted price
Larger absolute value → stronger influence

---

## 9. Saving the Final Model

The best full pipeline model was saved using:

```python
joblib.dump(best_model, "best_house_price_model.pkl")
```

This file includes:

* Imputer
* Scaler
* Feature selector
* Ridge model
* Tuned hyperparameters

Can be loaded for production or further inference:

```python
model = joblib.load("best_house_price_model.pkl")
preds = model.predict(new_data)
```

---

# Project Structure (Suggested)

```
│── README.md
│── notebook.ipynb
│── best_house_price_model.pkl
│── requirements.txt
```

---

# Technologies Used

* Python
* scikit-learn
* pandas
* numpy
* matplotlib / seaborn
* joblib

---

# How to Run

1. Clone repository
2. Install dependencies
3. Open notebook
4. Run all cells
5. Saved model will be generated as `.pkl`

---
