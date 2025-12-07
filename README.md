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
