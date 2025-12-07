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

# Machine Learning Lab 3 — Deep Neural Network (Forward Pass, MNIST)

This repository contains my work for **Machine Learning Laboratory: Deep Neural Network**.  
The assignment focuses on building a **multilayer feedforward neural network** for **multi-class classification on MNIST** using **only NumPy** (for matrix operations) and **Matplotlib/Seaborn** (for visualization). The key constraint is to implement the **forward pass only**, without backpropagation, and evaluate performance using standard multi-class metrics. :contentReference[oaicite:0]{index=0}

## Objectives
- Implement the **forward pass** of a neural network with matrix operations.
- Apply activation functions based on textbook equations (Eq. 6.14–6.18).
- Use **softmax** and **cross-entropy loss** for multi-class classification (Eq. 6.36–6.37).
- Experiment with different layer sizes and activation choices.
- Evaluate using **accuracy**, **confusion matrix**, **ROC**, and **precision/recall/F1**. :contentReference[oaicite:1]{index=1}

## Key Rules / Constraints
- Use the **MNIST dataset provided by the TA**.
- Use **NumPy** for the model implementation.
- Use **Matplotlib/Seaborn** for plotting.
- **Forward pass only** (no backpropagation required in this lab).
- You are expected to follow the provided arithmetic instructions and template structure. :contentReference[oaicite:2]{index=2}

## What You Need to Implement

### Core Components
- **Activation functions**:
  - ReLU
  - tanh
  - softplus
  - leaky ReLU  
- **One-hot encoding**
- **Cross-entropy loss**
- **Softmax**
- **Forward pass** across multiple layers (Eq. 6.19). :contentReference[oaicite:3]{index=3}

### Evaluation Utilities
- Confusion matrix for 10 classes
- ROC computation for multi-class
- Metrics report:
  - TP, FP, FN, TN per class
  - precision, recall, F1
  - overall accuracy. :contentReference[oaicite:4]{index=4}

## Grading (Homework Assignment)
**Implementation (50%)**
1. (10%) Feedforward NN with **more than one hidden layer**
2. (10%) Activation functions + softmax + cross-entropy implemented from scratch
3. (15%) Model runs correctly and produces predictions
4. Evaluation:
   - (5%) Confusion matrix plot
   - (5%) ROC curves for 10 classes
   - (5%) Precision/Recall/F1/Overall Accuracy

**Conceptual Questions (20%)**
1. Explain your model design and improvements with before/after performance.
2. Analyze performance: which classes are harder and why.

**Submission is capped at 70% max for this homework section** per the handout. :contentReference[oaicite:5]{index=5}

## Dataset
MNIST is loaded from IDX files using the template approach:
- `train-images.idx3-ubyte__`
- `train-labels.idx1-ubyte__`
- `t10k-images.idx3-ubyte__`
- `t10k-labels.idx1-ubyte__`

The template suggests using a smaller subset (e.g., first 500 train / 200 test) and allows you to experiment with these counts. :contentReference[oaicite:6]{index=6}

## Suggested Repository Structure
```text
.
├─ 112101014_Lab3_Homework.ipynb
├─ 112101014_Lab3_Homework.pdf
├─ train-images.idx3-ubyte__
├─ train-labels.idx1-ubyte__
├─ t10k-images.idx3-ubyte__
├─ t10k-labels.idx1-ubyte__
└─ README.md
````

## Environment

Recommended:

* Python 3.x
* numpy
* matplotlib
* seaborn (optional, for confusion matrix heatmap)
* pandas (optional, per template)

Install:

```bash
pip install numpy matplotlib seaborn pandas
```

## How to Run

### Jupyter

Open:

* `112101014_Lab3_Homework.ipynb`

Run all cells to:

* initialize random weights
* perform forward passes
* compute loss/accuracy
* generate evaluation plots.

## Expected Outputs

Your notebook/script should produce:

* Training loop logs across epochs
* Best loss/accuracy summary
* Test loss/accuracy
* **Confusion matrix** (plotted)
* **ROC curves** for 10 classes
* Printed **classification report**.

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab3_Homework.pdf`

   * Answer conceptual questions
   * Include result screenshots in the last pages of the provided PDF
2. **Code**: `StudentID_Lab3_Homework.py` or `.ipynb`

Deadline:

* **Sunday – 21:00 PM** (per handout). 

## Academic Integrity

Plagiarism is strictly prohibited. Submitting copied work may result in penalties. 

# Machine Learning Lab 3 (In-Class) — Deep Neural Network Forward Pass

This repository contains my **Lab 3 In-Class Assignment** for the Machine Learning laboratory on **Deep Neural Networks**.  
The focus of this in-class task is to implement a **1-hidden-layer feedforward neural network forward pass** from scratch using **NumPy**, integrate and compare multiple **activation functions**, and manually verify one full forward-pass computation. :contentReference[oaicite:0]{index=0}

## Objectives
- Understand the architecture and flow of a simple feedforward neural network with **one hidden layer**.
- Manually implement and compare activation functions:
  - **tanh**
  - **hard tanh**
  - **softplus**
  - **ReLU**
  - **leaky ReLU**
- Perform **forward propagation** using matrix operations in NumPy.
- Analyze how different activation functions affect hidden layer outputs and final predictions. :contentReference[oaicite:1]{index=1}

## What I Implemented
### 1. Activation Functions
Implemented using their mathematical definitions:
- `tanh(x)`
- `hard_tanh(x)`
- `softplus(x)`
- `relu(x)`
- `leaky_relu(x, alpha=0.1)` :contentReference[oaicite:2]{index=2}

### 2. One Hidden Layer Forward Pass
Based on the provided equations and fixed weights:
- Construct input with bias term.
- Compute hidden pre-activation `a1`.
- Apply chosen activation to obtain `z1`.
- Add hidden bias node `z0 = 1`.
- Compute output `y` using `W2`. :contentReference[oaicite:3]{index=3}

### 3. Activation Comparison
- Ran the same forward pass with each activation function.
- Compared **hidden layer outputs** (`z1`) across activations.
- Captured screenshots for the report. :contentReference[oaicite:4]{index=4}

### 4. Manual Verification
- Selected one activation function (e.g., tanh).
- Manually computed:
  - hidden activations
  - augmented hidden vector with bias
  - final output  
  and matched these with code results. :contentReference[oaicite:5]{index=5}

## Grading Items (In-Class Assignment)
This in-class assignment contributes up to **30% max**:
1. (7.5%) Implement a feedforward NN with at least 1 hidden layer.
2. (10%) Integrate and evaluate the five activation functions.
3. (5%) Compare hidden layer outputs for each activation (screenshots).
4. (7.5%) Manually calculate the network output for one activation. :contentReference[oaicite:6]{index=6}

## Files
```text
.
├─ 112101014_Lab3_InClass.ipynb
├─ 112101014_Lab3_InClass.pdf
└─ README.md
````

## Environment

Suggested setup:

* Python 3.x
* numpy

Install dependencies:

```bash
pip install numpy
```

## How to Run

Open the notebook:

```bash
jupyter notebook 112101014_Lab3_InClass.ipynb
```

Then:

1. Choose an activation function in the code.
2. Run the forward pass cells.
3. Observe and record `a1`, `z1`, `z1_aug`, and `y`.

## Submission

Upload both to E3:

* **Report**: `StudentID_Lab3_InClass.pdf`
* **Code**: `StudentID_Lab3_InClass.py` or `StudentID_Lab3_InClass.ipynb`

Deadline is specified in the handout. Plagiarism is strictly prohibited. 

# Machine Learning Lab 4 (Homework) — Gradient Descent Optimizers (MNIST Binary Classification)

This repository contains my implementation for **Machine Learning Laboratory: Gradient Descent Homework**.

The goal of this lab is to implement **mini-batch SGD** and its extensions — **Momentum**, **Nesterov Momentum**, and **Adam** — **from scratch using NumPy**, then apply them to a **binary MNIST classification** task and compare their behavior through accuracy and misclassified samples.

---

## Objectives
- Implement **Mini-batch SGD (Algorithm 7.2)**.
- Extend to:
  - **SGD with Momentum (Algorithm 7.3)**
  - **SGD with Nesterov Momentum (Eq. 7.34)**
  - **Adam Optimizer (Algorithm 7.4)**
- Build a **binary classifier** on MNIST:
  - “Is this the target digit or not?”
- Compare optimizers using:
  - **Test accuracy**
  - **At least 5 misclassified samples** with true/predicted labels.

---

## Key Rules / Constraints
- **NumPy-only** implementation for ML logic.
- **Do not use** external ML libraries such as **scikit-learn**, **PyTorch**, etc.
- You may implement all optimizers in:
  - one file with selectable modes, or
  - separate files.

---

## Target Digit
Use the **last digit of your student ID** as:

```python
TARGET_DIGIT = <last_digit_of_student_id>
````

Convert labels to binary:

* `1` if label == TARGET_DIGIT
* `0` otherwise

---

## Dataset

MNIST is provided in IDX format:

* `train-images.idx3-ubyte__`
* `train-labels.idx1-ubyte__`
* `t10k-images.idx3-ubyte__`
* `t10k-labels.idx1-ubyte__`

Images should be normalized to `[0,1]`. A bias term is added to features before training.

---

## What I Implemented

### 1. Core Functions

* `sigmoid(z)`
* binary prediction with threshold (e.g., 0.5)
* accuracy evaluation

### 2. Optimizers

* **Mini-batch SGD**
* **SGD + Momentum**
* **SGD + Nesterov Momentum**
* **Adam**

Each optimizer:

* trains a linear binary classifier with sigmoid output
* logs training status (loss/accuracy)
* evaluates test accuracy
* shows misclassified samples

### 3. Misclassification Visualization

Displays at least 5 incorrect test predictions:

* reshaped 28×28 images
* annotated with:

  * `T:` true label (binary)
  * `P:` predicted label (binary)

---

## Grading Breakdown (Homework — 70% Max)

### Implementation (50%)

1. Correctly implemented, runnable, and shows accuracy + misclassified samples:

   * (15%) Mini-batch SGD
   * (10%) SGD with Momentum
   * (5%)  SGD with Nesterov Momentum
   * (15%) Adam
2. (5%) Compare accuracy and misclassified samples across methods

### Conceptual Questions (20%)

1. Which optimizer gave the best test accuracy? Why?
2. Differences in stability, convergence speed, or misclassification types
3. Effects of learning rate, batch size, and momentum

---

## Suggested Repository Structure

```text
.
├─ 112101014_Lab4_Homework.ipynb
├─ 112101014_Lab4_Homework.pdf
├─ train-images.idx3-ubyte__
├─ train-labels.idx1-ubyte__
├─ t10k-images.idx3-ubyte__
├─ t10k-labels.idx1-ubyte__
└─ README.md
```

---

## Environment

Recommended:

* Python 3.x
* numpy
* matplotlib

Install:

```bash
pip install numpy matplotlib
```

---

## How to Run

### Jupyter Notebook

Open and run:

```bash
jupyter notebook 112101014_Lab4_Homework.ipynb
```

### (Optional) Python Script

If you export a script:

```bash
python 112101014_Lab4_Homework.py
```

---

## Expected Outputs

For each optimizer:

* Training logs (loss/accuracy)
* **Final test accuracy**
* **Misclassified sample visualization**
* A short comparison summary

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab4_Homework.pdf`
2. **Code**: `StudentID_Lab4_Homework.py` or `StudentID_Lab4_Homework.ipynb`

---

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your work is original.

---

## Notes

This README follows the requirements and rubric described in the provided Lab 4 homework handout. 

