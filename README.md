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

# Machine Learning Lab 4 (In-Class) — Stochastic Gradient Descent (SGD) on MNIST Binary Classification

This repository contains my **Lab 4 In-Class** work for the Machine Learning laboratory on **Gradient Descent**.  
In this assignment, I implement **Stochastic Gradient Descent (SGD)** using **one sample at a time** (Algorithm 7.1), apply the **sigmoid** function for **binary classification**, and evaluate performance on **MNIST** with **accuracy** and **misclassified samples visualization**, using **NumPy only** for model computations. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Understand and implement **SGD (one-sample update)** based on Algorithm 7.1.
- Perform binary classification using the **sigmoid** activation.
- Manually update model weights using the gradient of the loss.
- Practice NumPy-based computation on real MNIST data.
- Evaluate results using:
  - **Test accuracy**
  - **Visualization of misclassified samples**. :contentReference[oaicite:1]{index=1}

---

## Task Overview
You will complete the provided template to build a binary classifier answering:

> **“Is this target digit or not?”**

Key steps include:
1. Load MNIST IDX files.
2. Implement:
   - `sigmoid(z)`
   - `sgd_logistic(X, y, eta, max_iters)`
3. Choose `TARGET_DIGIT`.
4. Convert labels into binary:
   - `1` if label == `TARGET_DIGIT`
   - `0` otherwise
5. Add a bias term to input features.
6. Train with SGD.
7. Predict probabilities and compute accuracy.
8. Show misclassified test images. :contentReference[oaicite:2]{index=2}

---

## Important Rules
- Use **only NumPy** for computations.
- **Do not** use scikit-learn, PyTorch, TensorFlow, etc.
- Follow the in-class template structure. :contentReference[oaicite:3]{index=3}

---

## Target Digit Rule
Set the binary target class to the **last digit of your student ID**.  
For example, if your ID ends in **4**, then:
```python
TARGET_DIGIT = 4
````

This is a graded requirement. 

---

## Dataset

MNIST is provided in IDX format:

* `train-images.idx3-ubyte__`
* `train-labels.idx1-ubyte__`
* `t10k-images.idx3-ubyte__`
* `t10k-labels.idx1-ubyte__`

Images are flattened and normalized to `[0, 1]` in the template. 

---

## Suggested Repository Structure

```text
.
├─ 112101014_Lab4_InClass.ipynb
├─ 112101014_Lab4_InClass.pdf
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

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook 112101014_Lab4_InClass.ipynb
```

### Python Script (if exported)

```bash
python 112101014_Lab4_InClass.py
```

---

## Expected Outputs

Your run should display:

* **Test accuracy** (printed)
* **At least 5 misclassified test images** with:

  * `T:` true label
  * `P:` predicted label
    as illustrated in the example output section of the handout. 

---

## Grading (In-Class — 30% Max)

Implementation:

1. **(15%)** Implement SGD based on Algorithm 7.1.
2. **(10%)** Model runs successfully, trains on MNIST, and outputs test accuracy.
3. **(5%)** Set `TARGET_DIGIT` to the last digit of your student ID and display misclassified samples (≥5). 

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab4_InClass.pdf`

   * Answer all conceptual questions.
   * Include result screenshots in the last pages of the provided PDF.
2. **Code**: `StudentID_Lab4_InClass.py` or `StudentID_Lab4_InClass.ipynb`

Deadline: **16:20 PM**. 

---

## Academic Integrity

Plagiarism is strictly prohibited. Submitting copied work will result in penalties. 

# Machine Learning Lab 5 (Homework) — Regularization (Early Stopping & Weight Decay)

This repository contains my implementation for **Machine Learning Laboratory: Regularization Homework**.

The assignment focuses on preventing overfitting by implementing and comparing:
- **Early Stopping (validation-based regularization)**
- **Weight Decay (L2 regularization)**

These methods are applied to the **MNIST** task using an **existing MLP structure** with:
- **Single hidden layer + ReLU**
- **Softmax output** for multi-class classification (with evaluation later aligned to the assignment’s setup). :contentReference[oaicite:0]{index=0}

---

## Objectives
- Understand why regularization is important for generalization.
- Implement **early stopping** and **L2 weight decay** in a NumPy-based MLP.
- Visualize and interpret:
  - training vs validation **loss curves**
  - training vs validation **accuracy curves**
- Compare model behavior and inspect misclassified samples. :contentReference[oaicite:1]{index=1}

---

## Key Rules / Constraints
- Use the provided neural network structure.
- **Do not use external ML libraries** (e.g., scikit-learn, PyTorch). :contentReference[oaicite:2]{index=2}
- Implement the required utilities and training logic yourself using **NumPy**.
- You may organize code as:
  - a single file with selectable modes, or
  - separate files for each method. :contentReference[oaicite:3]{index=3}

---

## What I Implemented

### Utilities
- `shuffle_numpy(X, y)`
- `split_train_val(X, y, val_ratio=0.2)`
- `one_hot(y, num_classes)`
- `accuracy(Y_pred, Y_true)` :contentReference[oaicite:4]{index=4}

### Model (MLP)
- Forward:
  - `ReLU` hidden activation
  - `Softmax` output
- Loss:
  - Cross-entropy
  - **+ L2 penalty** when weight decay is enabled
- Backward:
  - Gradient updates including **λ * W** terms for L2 regularization. :contentReference[oaicite:5]{index=5}

### Training
- Baseline (no regularization)
- **Early Stopping**
  - Stop if validation loss doesn’t improve for `patience` epochs
- **Weight Decay**
  - Experiment with **3 different λ values**
- (Optional/Bonus if you tried) early stopping + weight decay together

### Plotting
- Training vs validation loss curves
- Training vs validation accuracy curves
for each method. :contentReference[oaicite:6]{index=6}

---

## Grading Breakdown (70% Max)

### Implementation (50%)
- **(20%) Early Stopping**
- **(20%) Weight Decay (L2)**
- **(10%) Comparison**
  - Curves + brief discussion
  - Results with **3 different λ values** :contentReference[oaicite:7]{index=7}

### Questions (20%)
1. Which method gave best test accuracy and why?
2. Compare train/val loss trends for overfitting/underfitting evidence.
3. How did λ or patience affect performance? :contentReference[oaicite:8]{index=8}

---

## Dataset
MNIST in IDX format:
- `train-images.idx3-ubyte__`
- `train-labels.idx1-ubyte__`

(And corresponding test files if included in your workspace.)  
Images are normalized to `[0, 1]` following the template. :contentReference[oaicite:9]{index=9}

---

## Suggested Repository Structure
```text
.
├─ 112101014_Lab5_Homework.ipynb
├─ 112101014_Lab5_Homework.pdf
├─ train-images.idx3-ubyte__
├─ train-labels.idx1-ubyte__
├─ t10k-images.idx3-ubyte__            (if provided)
├─ t10k-labels.idx1-ubyte__            (if provided)
└─ README.md
````

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

```bash
jupyter notebook 112101014_Lab5_Homework.ipynb
```

Run all cells to:

1. load and split data
2. train baseline
3. train early stopping model
4. train weight decay models with multiple λ
5. generate curves and comparisons

---

## Expected Outputs

* Printed training logs
* Plots:

  * Loss curves (train vs val)
  * Accuracy curves (train vs val)
* Comparison discussion
* Results for **3 λ values** for weight decay. 

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab5_Homework.pdf`
2. **Code**: `StudentID_Lab5_Homework.py` or `.ipynb`

Deadline:

* **Sunday, 21:00 PM**

Plagiarism is strictly prohibited. 

# Machine Learning Lab 5 (In-Class) — Backpropagation Autodiff (Logistic Classifier on MNIST)

This repository contains my **Lab 5 In-Class Assignment** for the Machine Learning laboratory on **Backpropagation**.  
The main goal is to integrate a simple **automatic differentiation (autodiff)** mechanism (both **forward-mode** and **reverse-mode/backprop-style**) into an **SGD-trained logistic classifier** for **MNIST binary classification**. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Understand the core concept of **backpropagation** in neural network training.
- Simulate and visualize:
  - **Primal values** through the forward pass
  - **Forward-mode autodiff** (tangent propagation)
  - **Reverse-mode autodiff** (adjoint/backprop propagation)
- Interpret gradient flow using a **computational trace table** and bar-chart visualization.
- Implement everything using **NumPy only**. :contentReference[oaicite:1]{index=1}

---

## Task Overview
You will complete the provided in-class template to:

1. **Load MNIST** from IDX files.
2. Create a binary task:
   - “**Is it the target digit or not?**”
3. Implement/complete:
   - `sigmoid(z)` *(optional per template)*
   - `sigmoid_derivative(z)`
   - `trace_autodiff_example(x1, x2)`  
     - Produces a table containing:
       - Primal values
       - Forward tangents
       - Reverse adjoints
   - `your_sgd_logistic(X, y, eta, max_iters)`  
     - Use last week’s SGD structure
     - Integrate autodiff tracing at iteration 0
4. **Predict**, compute **test accuracy**.
5. **Show misclassified samples**.
6. **Plot autodiff trace graphs**. :contentReference[oaicite:2]{index=2}

---

## Key Rules / Constraints
- **Use only NumPy** for computations.
- **Do not use** scikit-learn, PyTorch, TensorFlow, etc.
- Follow the in-class template and output format. :contentReference[oaicite:3]{index=3}

---

## Target Digit Requirement
Set the binary class to the **last digit of your student ID**.  
Example:
```python
TARGET_DIGIT = 7
````

Label conversion:

````python
y_train_bin = np.where(y_train == TARGET_DIGIT, 1, 0)
y_test_bin  = np.where(y_test  == TARGET_DIGIT, 1, 0)
``` :contentReference[oaicite:4]{index=4}

---

## Dataset
MNIST IDX files:
- `train-images.idx3-ubyte__`
- `train-labels.idx1-ubyte__`
- `t10k-images.idx3-ubyte__`
- `t10k-labels.idx1-ubyte__`

Images are normalized to `[0, 1]`, and a **bias column** is added to `X`. :contentReference[oaicite:5]{index=5}

---

## Grading (In-Class — 30% Max)

### Implementation
1. **(10%)** Implement the backpropagation autodiff.
2. **(10%)** Code runs successfully and outputs **primal/forward/reverse** traces.
3. **(5%)** Correct `TARGET_DIGIT` rule and display results similar to the example output.
4. **(5%)** Brief discussion explaining the graphs and results. :contentReference[oaicite:6]{index=6}

---

## Suggested Repository Structure
```text
.
├─ 112101014_Lab5_InClass.ipynb
├─ 112101014_Lab5_InClass.pdf
├─ train-images.idx3-ubyte__
├─ train-labels.idx1-ubyte__
├─ t10k-images.idx3-ubyte__
├─ t10k-labels.idx1-ubyte__
└─ README.md
````

---

## Environment

Recommended:

* Python 3.x
* numpy
* matplotlib
* pandas *(for trace table display, as suggested by template)*

Install:

```bash
pip install numpy matplotlib pandas
```

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook 112101014_Lab5_InClass.ipynb
```

Run all cells to:

* train the logistic classifier with SGD
* generate the autodiff trace table
* plot primal/forward/reverse bar charts
* show misclassified samples

---

## Expected Outputs

Your results should include:

* Printed **test accuracy** for the binary task
* A displayed **misclassified samples** figure
* An **Autodiff Trace Table** (sample features)
* Three bar charts:

  1. **Primal Values**
  2. **Forward-Mode Autodiff**
  3. **Reverse-Mode Autodiff** 

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab5_InClass.pdf`

   * Add screenshots and short discussion in the last pages of the provided PDF.
2. **Code**: `StudentID_Lab5_InClass.py` or `StudentID_Lab5_InClass.ipynb`

Deadline: **16:20 PM**. 

---

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your work is original. 

# Machine Learning Lab 6 (Homework) — CNN Cats vs Dogs (Keras/TensorFlow)

This repository contains my implementation for **Machine Learning Laboratory: CNN Homework**.  
The goal of this assignment is to **train a Convolutional Neural Network (CNN) from scratch** to classify **cats vs. dogs** using the provided dataset and starter template, then **improve validation accuracy** by redesigning and fine-tuning the architecture. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Train a CNN for **binary image classification** (cat = 0, dog = 1).
- Start from the baseline template model (which is intentionally not strong).
- Improve performance by:
  - modifying CNN depth/width and layer configs
  - adding **data augmentation**
  - applying regularization such as **Dropout** and **BatchNormalization**
  - exploring custom pooling/structure ideas  
  while keeping the model **fully self-designed**. :contentReference[oaicite:1]{index=1}

---

## Key Rules / Constraints
- **No transfer learning.**
- **Do not use pre-trained models** like VGG, ResNet, EfficientNet, etc.  
  (No imports from `keras.applications`.) :contentReference[oaicite:2]{index=2}
- Your model must be trained **from scratch** using Keras/TensorFlow layers.
- The report must briefly explain your redesigned model and performance gains. :contentReference[oaicite:3]{index=3}

---

## Dataset
The dataset is a cats vs dogs image set (~25,000 images) loaded via the template’s Google Drive download step using `gdown`.  
Images are resized and normalized to `[0, 1]`.  
The template uses a `tf.data.Dataset` pipeline with shuffling, batching, and a train/test split by `take()` and `skip()`. :contentReference[oaicite:4]{index=4}

---

## Baseline Model (Template)
The provided baseline is a simple Sequential CNN with stacked Conv2D + MaxPooling blocks and a Dense classifier ending with:
- `Dense(1, activation="sigmoid")`
- `loss="binary_crossentropy"`
- `optimizer=Adam`  

This baseline is meant as a starting point and **does not achieve high accuracy**. :contentReference[oaicite:5]{index=5}

---

## What I Changed (Summary)
In my improved version, I focused on:
- **Deeper feature extraction** with more structured convolutional blocks.
- **Regularization** to reduce overfitting:
  - Dropout
  - BatchNormalization
- **Data augmentation** to improve generalization.
- **Hyperparameter tuning** (learning rate, batch size, epochs) and optional callbacks
  such as learning-rate reduction and early stopping.

(Details and model summary are included in the homework report.) :contentReference[oaicite:6]{index=6}

---

## Evaluation Outputs
The assignment requires including:
- **Model summary**
- **Training/validation accuracy & loss plots**
- **Confusion matrix**
- **Classification report**  

These should be captured and placed in the last pages of the provided PDF report template. :contentReference[oaicite:7]{index=7}

---

## Grading Breakdown (Homework — 70%)
**Implementation (45%)**
1. (5%) Fine-tuning to achieve better accuracy  
2. (20%) Model redesign with advanced techniques  
   (e.g., augmentation, Dropout, BatchNorm, self-designed structures)
3. (15%) Demonstrate significantly improved **validation accuracy** vs baseline  
4. (5%) Include all required evaluation results  

**Questions (25%)**
5. Architecture changes & rationale  
6. Overfitting reduction methods & evidence from curves  
7. Error analysis from the confusion matrix and improvement ideas :contentReference[oaicite:8]{index=8}

---

## Repository Structure (Suggested)
```text
.
├─ 112101014_Lab6_Homework.ipynb
├─ 112101014_Lab6_Homework.pdf
├─ student_ID.keras                 # saved trained model
└─ README.md
````

---

## Environment

Recommended:

* Python 3.x
* TensorFlow / Keras
* numpy
* pandas
* matplotlib
* seaborn (for confusion matrix visualization)

Install (example):

```bash
pip install tensorflow numpy pandas matplotlib seaborn gdown
```

---

## How to Run

### Jupyter Notebook

Open and run:

```bash
jupyter notebook 112101014_Lab6_Homework.ipynb
```

The notebook will:

1. Download and unzip the dataset.
2. Build the baseline CNN.
3. Train and validate.
4. Run your improved architecture.
5. Save the model as:

```python
model.save("student_ID.keras")
```

6. Generate plots and evaluation reports. 

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab6_Homework.pdf`
2. **Code**: `StudentID_Lab6_Homework.py` or `StudentID_Lab6_Homework.ipynb`

Deadline: **16:20 PM**.
Plagiarism is strictly prohibited. 

---

## Notes

* Focus on **clear before/after comparison** in your report:
  baseline vs improved validation accuracy,
  plus evidence from training/validation curves.
* If accuracy improves but overfitting appears,
  strengthen augmentation/regularization or adjust learning rate/epochs. 
