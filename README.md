# 🏠 California Housing Price Prediction
### Linear Regression with Gradient Descent — Built from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=flat-square&logo=numpy)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()

> 🔧 **No scikit-learn for modelling.** The linear regression model and gradient descent optimizer are implemented entirely from scratch using NumPy.

---

## 📌 About This Project

This is my **first machine learning project**, where I predict California housing prices using a **Linear Regression model built from the ground up**. Rather than using a ready-made ML library for the model, I manually implemented the **cost function**, **gradient computation**, and **parameter update loop** — giving me a deep understanding of how linear regression actually works under the hood.

The model is trained on the [California Housing Dataset](https://raw.githubusercontent.com/akmand/datasets/main/california_housing.csv) and predicts the **median house value** for California districts using engineered features derived from census data.

---

## 📂 Project Structure

```
first-project/
│
└── california_linear_reg.ipynb   # Full notebook: EDA, preprocessing, training, evaluation
```

---

## 🗂️ Dataset

- **Source:** [akmand/datasets on GitHub](https://raw.githubusercontent.com/akmand/datasets/main/california_housing.csv)
- **Origin:** 1990 U.S. Census, California block groups
- **Size:** ~20,640 records

### Raw Features

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude of the district |
| `latitude` | Geographic latitude of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms in the district |
| `total_bedrooms` | Total number of bedrooms *(had missing values)* |
| `population` | Total population of the district |
| `households` | Total number of households |
| `median_income` | Median household income |
| `ocean_proximity` | Categorical: how close the district is to the ocean |
| `median_house_value` | 🎯 **Target variable** — median house value (USD) |

---

## 🔬 What I Did — Step by Step

### 1. 📥 Data Loading
- Loaded the dataset directly from a public URL using Pandas
- Inspected the first few rows with `.head()` to understand the structure

### 2. 🧹 Data Preprocessing

**Missing Value Imputation:**
- `total_bedrooms` had missing values — filled them with the **column median** to avoid data loss without introducing bias

**Feature Scaling (Z-Score Normalization):**
- Wrote a custom `apply_z_score()` function from scratch:
  ```python
  def apply_z_score(data):
      mean = np.mean(data)
      std  = np.std(data)
      return (data - mean) / std
  ```
- Applied it to all 11 numerical features so gradient descent converges smoothly (features on vastly different scales would cause it to diverge or crawl)

**One-Hot Encoding:**
- Used `pd.get_dummies()` to convert the categorical `ocean_proximity` column into binary indicator columns (e.g., `INLAND`, `NEAR BAY`, `<1H OCEAN`, etc.)

**Bias Term:**
- Prepended a column of ones to the feature matrix to act as the **intercept/bias** term in the linear model

### 3. ⚙️ Feature Engineering

Beyond the raw features, I created **6 new derived features** to help the model capture richer signals:

| Engineered Feature | Formula | Why it Matters |
|---|---|---|
| `rooms_per_hh` | `total_rooms / households` | Average rooms per household — captures housing spaciousness |
| `bedr_per_room` | `total_bedrooms / total_rooms` | Bedroom ratio — reflects the type of housing in the area |
| `pop_per_hh` | `population / households` | Average occupants per household — reflects crowding |
| `inc2` | `median_income²` | Captures non-linear income effects |
| `lat_lon` | `latitude × longitude` | Interaction term to encode combined geographic location |
| `inc_lat` | `median_income × latitude` | Income-location interaction |

### 4. 🔀 Train / Test Split
- Shuffled the entire dataset with `np.random.permutation` to remove any ordering effects
- Split into **80% training** (~16,512 samples) and **20% testing** (~4,128 samples)

### 5. 📐 Target Transformation
- Applied a **log transformation** to the target variable before training:
  ```python
  Y_full = np.log(Y_full)
  ```
- House prices are **right-skewed** — a few very expensive houses stretch the distribution. Taking the log compresses this tail, making the distribution more symmetric and reducing the outsized influence of outliers on the loss function.

### 6. 🤖 Linear Regression — Built from Scratch

The model is defined by:

```
ŷ = X · w
```

Where `w` is the weight vector learned via **Batch Gradient Descent**.

**Cost Function (Mean Squared Error):**
```
J(w) = (1 / 2m) × (Xw − Y)ᵀ(Xw − Y)
```

**Gradient:**
```
∂J/∂w = (1 / m) × Xᵀ(Xw − Y)
```

**Weight Update Rule:**
```
w ← w − (lr / m) × Xᵀ(Xw − Y)
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Learning Rate (`lr`) | `0.01` |
| Iterations | `10,000` |
| Initial Weights | All zeros |

Full training loop:
```python
w = np.zeros((X.shape[1], 1))
for i in range(10000):
    p    = np.dot(X, w)
    er   = p - Y
    cost = np.dot(er.T, er).item() / (2 * sp)
    c.append(cost)
    w    = w - (lr / sp) * np.dot(X.T, er)
```

### 7. 📊 Evaluation

A custom `pred()` function reverses the log-transform on both predictions and actuals before computing **Mean Absolute Percentage Error (MAPE)**:

```python
def pred(x, y):
    p  = np.exp(np.dot(x, w))    # reverse log transform on predictions
    y  = np.exp(y)                # reverse log transform on actuals
    er = Σ |pᵢ − yᵢ| / yᵢ × 100 / n
    return er
```

---

## 📈 Results

| Set | MAPE |
|---|---|
| **Training Set** | **25.90%** |
| **Test Set** | **25.75%** |

The nearly identical training and test errors confirm that the model **generalises well** and is not overfitting.

### Cost Convergence

The cost function drops sharply in the first ~200 iterations and then smoothly plateaus — confirming that gradient descent **converged successfully**.

![Cost Convergence Curve](cost_curve.png)

> 💡 A MAPE of ~25% means predictions are off by roughly one quarter of the actual house price on average. This is a solid baseline for a pure linear model on a non-linear real-world dataset. Advanced models like Random Forest typically achieve ~15% MAPE on this same data.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `Python 3.x` | Core programming language |
| `Jupyter Notebook` | Interactive development environment |
| `NumPy` | Matrix math, gradient descent, custom Z-score normalization |
| `Pandas` | Data loading, missing value handling, one-hot encoding |
| `Matplotlib` | Cost convergence visualization |

> ✅ No ML library (e.g., scikit-learn) was used for the model — everything from the cost function to the weight updates is written by hand.

---


## 💡 Key Learnings

Building this project from scratch gave me deep, hands-on understanding of:

- **Why feature scaling is essential** — without Z-score normalization, gradient descent diverges or fails to converge in reasonable time
- **How gradient descent works** — computing the gradient, choosing a learning rate, and watching cost converge over iterations
- **Why log-transforming skewed targets helps** — housing prices are not normally distributed; the log transform leads to a better-behaved loss surface
- **The power of feature engineering** — derived ratios like `rooms_per_hh` and `bedr_per_room` carry more predictive signal than raw counts
- **How to evaluate regression models** — using MAPE as a percentage-based, human-interpretable error metric
- **The full ML pipeline from scratch** — Load → Clean → Engineer → Scale → Split → Train → Evaluate

---

## 🚀 Future Improvements

- [ ] Add the interaction features (`inc_lat`, `lat_lon`, `inc2`) into the final model and measure their impact
- [ ] Experiment with different learning rates and compare convergence behaviour
- [ ] Implement **Ridge (L2) Regularisation** to penalise large weights and handle correlated features
- [ ] Try **Polynomial Features** to capture non-linear relationships
- [ ] Compare performance against a Random Forest or XGBoost baseline
- [ ] Add a **geographic scatter plot** of California coloured by actual vs predicted prices
- [ ] Deploy as an interactive web app using **Streamlit**

---

## 📬 Connect With Me

- **GitHub:** [@amithgoud](https://github.com/amithgoud)

---

> ⭐ *If you found this project interesting, consider giving it a star — it means a lot for a first project!*
