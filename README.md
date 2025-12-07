# -Machine-Learning

# Machine Learning Lab 1 — Regression

This repository contains my implementation for **Machine Learning Laboratory: Regression**.  
The lab focuses on understanding and implementing **linear regression** and **ridge regression** from scratch, training with **gradient descent**, evaluating with **MSE**, and extending to a **closed-form ridge solution** and **predictive distribution with confidence intervals**.  
(Assignment details summarized from the provided lab handout.)  

## Objectives
- Understand key concepts and math behind regression models.
- Implement and evaluate regression models **from scratch using NumPy**.
- Connect regression with least squares, likelihood estimation, and the bias–variance trade-off.
- Train models using **gradient descent** and analyze learning behavior.

## Key Rules
- **Do not use scikit-learn’s built-in regression** (e.g., `sklearn.linear_model.LinearRegression`).
- Use **NumPy** for core implementation.
- Use **Matplotlib** for visualization.

## Tasks & Grading Breakdown
Hands-on + assignment items:
1. **Standard Linear Regression (Gradient Descent)** — 10%  
   - Compute gradients for weight and bias  
   - Update parameters  
   - Report learned parameters
2. **Evaluation with MSE** — 10%  
   - Implement `compute_mse()`  
   - Report test MSE
3. **Ridge Regression (Gradient Descent, L2 Regularization)** — 10%  
   - Add L2 term to loss  
   - Derive/implement updated gradients  
   - Report parameters and MSE
4. **Plot Training Loss Curve** — 10%  
   - Store loss per iteration  
   - Compare standard vs ridge  
   - Try different learning rates and iterations
5. **Closed-form Ridge Regression** — 20%  
   - Implement closed-form solution  
   - Report parameters and MSE
6. **Predictive Distribution** — 20%  
   - Implement predictive mean/variance  
   - Print example predictive means (first 5)
7. **Visualization of Predictions & Confidence Intervals** — 20%  
   - Scatter plots for ground truth and predictions  
   - Confidence interval shading with `fill_between`

## Dataset
The dataset is provided as:
- `regression_data.npy`  
  Contains `(x_train, x_test, y_train, y_test)`.

The lab handout points to the sample code/dataset source:
- https://github.com/Satriosnjya/ML-Labs.git

## Environment
Recommended:
- Python 3.x
- numpy
- matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
````

## Project Structure (Suggested)

```text
.
├─ 112101014_Lab1.ipynb
├─ 112101014_Lab1.pdf
├─ regression_data.npy
├─ README.md
└─ (optional) 112101014_Lab1.py
```

## How to Run

### Option 1: Jupyter Notebook

Open and run:

* `112101014_Lab1.ipynb`

### Option 2: Python Script (if you export one)

```bash
python 112101014_Lab1.py
```

## Expected Outputs

Your run should produce:

* Learned parameters for:

  * Standard GD regression
  * Ridge GD regression
  * Closed-form ridge regression
* MSE values for each method
* Loss curves
* Prediction scatter plot comparisons
* Predictive distribution plot with confidence intervals

## Report & Submission

You should submit:

1. **Report PDF** with result screenshots placed in the specified pages.
2. **Code** (`.py` or `.ipynb`).

Naming convention:

* Report: `StudentID_Lab1.pdf`
* Code: `StudentID_Lab1.py` or `StudentID_Lab1.ipynb`

Late policy and due time are specified in the handout.

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your implementation and report are your own work.

---

## Notes

This README is aligned with the lab handout requirements and task list. 

# Machine Learning Lab 2 — Classification (Logistic & Probit Regression)

This repository contains my implementation for **Machine Learning Laboratory: Logistic & Probit Regression**.  
Although the handout header mentions “Regression,” this lab is a **binary classification** task using **logistic regression** (sigmoid) and **probit regression** (normal CDF), implemented **from scratch with NumPy**, trained by **gradient descent**, and evaluated with **confusion matrix**, **ROC**, and **AUC**. :contentReference[oaicite:0]{index=0}

## Objectives
- Develop a deeper understanding of **classification models**, focusing on logistic and probit regression.
- Learn the mathematical foundations and differences between:
  - **Sigmoid** (logistic activation)
  - **Normal CDF** (probit activation)
- Implement both models **from scratch using NumPy** without relying on sklearn’s built-in classifiers.
- Train models using **gradient descent**, analyze convergence, and evaluate performance.
- Apply and compare **confusion matrices**, **ROC curves**, and **AUC**. :contentReference[oaicite:1]{index=1}

## Problem Description
Goal: predict a binary label **y ∈ {0,1}** indicating whether an individual earns **more than 50K per year** based on demographic and work-related factors.  
Inputs are categorical features (e.g., `age_bin`, `occupation_bin`, `education_bin`, etc.) that should be **one-hot encoded** before training. :contentReference[oaicite:2]{index=2}

## Key Rules
- **Do NOT use** sklearn’s built-in classifiers (e.g., `sklearn.linear_model.LogisticRegression`).  
  You must implement training and prediction yourself using NumPy. :contentReference[oaicite:3]{index=3}
- Use **manual preprocessing**:
  - One-hot encoding for categorical bins
  - Feature scaling using training mean/std

## Tasks & Grading

### Assignment (70%)
1. **Implement Logistic Regression** (20%)  
   - Train with learning rate = **0.01**, iterations = **3000**. :contentReference[oaicite:4]{index=4}
2. **Implement Probit Regression** (20%)  
   - Use normal CDF (handout suggests an approximation for Φ). :contentReference[oaicite:5]{index=5}
3. **Compute Confusion Matrices** (10%)  
   - Report **TP, TN, FP, FN** for both models. :contentReference[oaicite:6]{index=6}
4. **Generate and Analyze ROC Curve** (10%)  
   - Compute **TPR/FPR** across thresholds; plot ROC. :contentReference[oaicite:7]{index=7}
5. **Compute and Compare AUC** (10%)  
   - Use numerical approximation (e.g., trapezoidal). :contentReference[oaicite:8]{index=8}

### Conceptual Questions (30%)
You will answer questions comparing:
- Confusion matrices
- ROC/AUC differences
- Effects of learning rate & iterations
- When to choose logistic vs probit
- How activation functions shape behavior
- Why these metrics matter for model selection :contentReference[oaicite:9]{index=9}

## Dataset
The handout indicates the dataset and sample code can be found in:
- `ML-Labs` GitHub repository
- You will likely use a CSV for classification and split by a `flag` column (`train` / `test`). :contentReference[oaicite:10]{index=10}

Typical preprocessing steps:
```python
cat_feats = [
    'age_bin','capital_gl_bin','education_bin',
    'hours_per_week_bin','msr_bin','occupation_bin','race_sex_bin'
]

# one-hot encode
x_train = pd.get_dummies(x_train, columns=cat_feats, drop_first=True)
x_test  = pd.get_dummies(x_test,  columns=cat_feats, drop_first=True)

# numpy conversion + manual scaling
mean_train = np.mean(x_train, axis=0)
std_train  = np.std(x_train, axis=0)
x_train_scaled = (x_train - mean_train) / std_train
x_test_scaled  = (x_test - mean_train) / std_train
````

(Structure based on the handout’s data preparation outline.) 

## Repository Structure (Suggested)

```text
.
├─ 112101014_Lab2.ipynb
├─ 112101014_Lab2.pdf
├─ (dataset files, e.g., classification.csv)
└─ README.md
```

## Environment

Recommended:

* Python 3.x
* numpy
* pandas
* matplotlib

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

## How to Run

### Option 1: Jupyter Notebook

Open and run:

* `112101014_Lab2.ipynb`

### Option 2: Export as Script (optional)

```bash
python 112101014_Lab2.py
```

## Expected Outputs

Your results should include:

* Learned weights/bias for **logistic** and **probit** models
* **Confusion matrices** for both approaches
* **ROC curves** (side-by-side comparison)
* **AUC** values
* A brief hyperparameter study showing how learning rate/iterations affect convergence and ROC/AUC 

## Report & Submission

Submit both:

1. **Report** answering all conceptual questions
2. **Code** (`.py` or `.ipynb`)

File naming:

* `StudentID_Lab2.pdf`
* `StudentID_Lab2.py` or `StudentID_Lab2.ipynb` 

Late policy:

* 1 day late: 10% deduction. 

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your implementation and report are your own work. 
