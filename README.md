# -Machine-Learning

# Machine Learning Labs & Final Project (NYCU 2025)

This repository collects my implementations for the **Machine Learning Laboratory series** and related coursework deliverables, including in-class exercises, homework notebooks, and (if included) the final project. This consolidated README is refined from my existing draft notes. :contentReference[oaicite:0]{index=0}

---

## Contents
- Lab 1 — Regression
- Lab 2 — Classification (Logistic & Probit Regression)
- Lab 3 — Deep Neural Network (Forward Pass)
  - In-Class
  - Homework
- Lab 4 — Gradient Descent Optimizers
  - In-Class (SGD)
  - Homework (Mini-batch SGD, Momentum, Nesterov, Adam)
- Lab 5 — Regularization & Backprop/Autodiff
  - In-Class (Autodiff demo)
  - Homework (Early Stopping & Weight Decay)
- Lab 6 — Convolutional Neural Networks
  - In-Class (2D Convolution + Edge Detection)
  - Homework (Cats vs Dogs CNN with Keras)
- Lab 7 — RNN / LSTM
  - In-Class (Vanilla RNN, manual updates)
  - Homework (Manual LSTM cell on MNIST)
- Lab 8 — Transformers & Generative Models
  - In-Class (GAN on MNIST)
  - Homework (Vision Transformer for 6-class defect classification)
- Lab 9 — Advanced GAN
  - Homework (GAN vs CycleGAN on FashionMNIST & CIFAR-10)
- Final Project — Metallic Surface Defect Detection (if present)

---

## General Notes
- Most labs follow the course rule of implementing core logic **from scratch** using **NumPy** or **PyTorch/TensorFlow** depending on the assignment.
- Each lab typically includes:
  - a notebook (`.ipynb`)
  - a report (`.pdf`)
  - optional exported script (`.py`)
- File names use the format:
  - `112101014_LabX_InClass.*`
  - `112101014_LabX_Homework.*`

---

## Lab Summaries

### Lab 1 — Regression
Focus:
- Linear regression & ridge regression
- Gradient descent training
- MSE evaluation
- Closed-form ridge solution
- Predictive distribution & confidence intervals

Typical outputs:
- learned parameters
- training loss curves
- prediction plots with confidence bands

---

### Lab 2 — Classification (Logistic & Probit)
Focus:
- Binary classification (income > 50K task)
- Logistic regression (sigmoid) vs probit regression (normal CDF)
- Manual one-hot encoding + feature scaling
- Confusion matrix, ROC curve, AUC

---

### Lab 3 — Deep Neural Network (Forward Pass)
Focus:
- Multi-layer feedforward NN for MNIST
- **Forward pass only** (no backprop in this lab)
- Multiple activation functions
- Softmax + cross-entropy
- Multi-class evaluation (confusion matrix, ROC, precision/recall/F1)

---

### Lab 4 — Gradient Descent Optimizers
**In-Class**
- SGD with one-sample updates for MNIST binary classification  
- Target digit = last digit of student ID

**Homework**
- Mini-batch SGD
- Momentum
- Nesterov Momentum
- Adam  
All implemented **from scratch using NumPy**, with accuracy comparison and misclassified sample visualization.

---

### Lab 5 — Regularization & Autodiff
**In-Class**
- Forward-mode vs reverse-mode autodiff tracing
- Integrate into a simple logistic classifier

**Homework**
- Early stopping
- Weight decay (L2)
- Train/val loss & accuracy curve comparisons
- Experiments with multiple λ values

---

### Lab 6 — CNN
**In-Class**
- Implement 2D convolution with padding/stride using NumPy
- Apply vertical/horizontal edge filters
- Binary visualization via thresholding

**Homework**
- Cats vs Dogs classification using Keras/TensorFlow
- CNN redesign to improve validation accuracy
- Data augmentation, Dropout, BatchNorm
- Confusion matrix + classification report

---

### Lab 7 — RNN / LSTM
**In-Class**
- Vanilla RNN for toy sentence sentiment classification
- Manual parameter updates (no built-in optimizers)

**Homework**
- Manual LSTM cell implementation
- MNIST treated as a 28-step sequence
- Hyperparameter tuning + test accuracy + example predictions

---

### Lab 8 — GAN & Vision Transformer
**In-Class**
- GAN on MNIST (single target digit)
- BCE loss, simple MLP G/D
- Save generated samples across epochs

**Homework**
- Vision Transformer from scratch
- 6-class defect dataset
- Manual patch embedding, positional encoding, MHA, MLP blocks
- Required: 70/30 train-test split, confusion matrix, classwise predictions

---

### Lab 9 — GAN vs CycleGAN
Focus:
- Extend GAN to **FashionMNIST** and **CIFAR-10**
- Train on **≥ 3 classes per dataset**
- Implement CycleGAN-style model for same classes
- **Mimic mode** (style targeting)
- Visual comparison of GAN vs CycleGAN results

---

## Final Project (If Included)
Focus:
- Metallic surface defect **object detection** (6 classes)
- Train an improved detector (pretrained backbone allowed, **must be modified**)
- Generate `submission.csv` for Kaggle evaluation

---

## Environment
Common dependencies across labs:
```bash
pip install numpy pandas matplotlib
````

Deep learning labs may also require:

```bash
pip install torch torchvision
pip install tensorflow
pip install einops
```

---

## Recommended Structure

```text
.
├─ Lab1/
├─ Lab2/
├─ Lab3/
├─ Lab4/
├─ Lab5/
├─ Lab6/
├─ Lab7/
├─ Lab8/
├─ Lab9/
├─ FinalProject/        # if applicable
└─ README.md
```

---

## Academic Integrity

All work in this repository is intended to follow course rules on **individual completion** and **plagiarism avoidance**.
Please do not copy or redistribute solutions in ways that violate the course policy.

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

```python
y_train_bin = np.where(y_train == TARGET_DIGIT, 1, 0)
y_test_bin  = np.where(y_test  == TARGET_DIGIT, 1, 0)
````

:contentReference[oaicite:4]{index=4}

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

# Machine Learning Lab 6 (In-Class) — CNN Ops from Scratch (NumPy)

This repository contains my **Lab 6 In-Class Assignment** for the Machine Learning laboratory on **Convolutional Neural Networks**.  
In this lab, I implement **basic 2D convolution (with padding and stride)** and apply **vertical/horizontal edge detection** on a grayscale image using **NumPy only**, then visualize results in **black & white** through thresholding. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Implement **2D convolution** and understand how **padding** and **stride** affect output size and information flow.
- Apply **vertical** and **horizontal** edge filters to detect structure patterns.
- Visualize:
  - Original image
  - Edge maps with **padding=1**
  - Edge maps with **stride=2**
- Convert outputs to **binary (black/white)** images via thresholding. :contentReference[oaicite:1]{index=1}

---

## Key Rules / Constraints
- Use **NumPy** for convolution computation.
- **Do not use high-level deep learning APIs** (e.g., PyTorch `nn.Conv2d`, `F.max_pool2d`, TensorFlow equivalents).
- **Do not use OpenCV built-in convolution** functions.  
  (OpenCV may be used only for image loading in the template.) :contentReference[oaicite:2]{index=2}

---

## Tasks

### 1. Load & Normalize Image
Load a grayscale test image (e.g., `original.png`) and normalize to `[0, 1]`. :contentReference[oaicite:3]{index=3}

### 2. Implement General 2D Convolution
Complete:
```python
def convolve2d(image, kernel, padding=0, stride=1):
    # 1) Flip kernel
    # 2) Apply zero-padding
    # 3) Compute output size
    # 4) Slide kernel with stride
    # 5) Sum element-wise products
````

The lab references the standard definition:
[
C(j,k) = \sum_l \sum_m I(j+l, k+m) K(l,m)
]


### 3. Define Edge Detection Filters

Fill in the **vertical** and **horizontal** 3×3 kernels provided in the slides/template. 

### 4. Apply Convolution

Compute:

* `vertical_edges` with `padding=1, stride=1`
* `horizontal_edges` with `padding=1, stride=1`
* `vertical_stride` with `padding=1, stride=2`
* `horizontal_stride` with `padding=1, stride=2` 

### 5. Binarize & Visualize

Use the provided `binarize()` helper to normalize and threshold outputs, then visualize **five views**:

1. Original
2. Vertical edges (pad=1)
3. Horizontal edges (pad=1)
4. Vertical edges (stride=2)
5. Horizontal edges (stride=2) 

---

## Grading (In-Class — 30% Max)

**Implementation**

1. (10%) Correct `convolve2d()` with kernel flipping, padding, stride
2. (5%) Correct application of vertical & horizontal filters
3. (5%) Correct binary thresholding and visualization of all five views

**Questions**
4. (5%) What patterns does the vertical edge filter detect vs horizontal?
5. (5%) What is the effect of padding and why use it? 

---

## Suggested Repository Structure

```text
.
├─ 112101014_Lab6_InClass.ipynb
├─ 112101014_Lab6_InClass.pdf
├─ original.png
└─ README.md
```

---

## Environment

Recommended:

* Python 3.x
* numpy
* matplotlib
* opencv-python *(for image loading only)*

Install:

```bash
pip install numpy matplotlib opencv-python
```

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook 112101014_Lab6_InClass.ipynb
```

Run all cells to generate:

* edge detection outputs (pad=1)
* strided outputs (stride=2)
* binarized black/white visualizations.

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab6_InClass.pdf`

   * Put screenshots of results in the last pages of the provided PDF.
2. **Code**: `StudentID_Lab6_InClass.py` or `StudentID_Lab6_InClass.ipynb`

Deadline: **16:20 PM**. 

---

## Academic Integrity

Plagiarism is strictly prohibited. 

# Machine Learning Lab 7 (Homework) — Manual LSTM Cell for MNIST

This repository contains my implementation for **Machine Learning Laboratory: LSTM Homework**.  
In this assignment, I build a **manual LSTM cell from scratch** and apply it to **MNIST digit classification** by treating each **28×28** image as a **sequence of 28 time steps** (each time step is a 28-dim row vector). :contentReference[oaicite:0]{index=0}

## Objectives
- Understand how **forget**, **input**, and **output** gates interact to update hidden and cell states.
- Implement the **LSTM forward pass manually** using the provided equations.
- Train the model in **PyTorch** while avoiding high-level RNN modules.
- Implement a **testing loop** to evaluate performance on the full MNIST test set.
- **Tune hyperparameters** to improve test accuracy.
- Visualize prediction results on example test images. :contentReference[oaicite:1]{index=1}

## Key Rules / Constraints
- **Do not use high-level RNN modules** such as `nn.LSTM`. :contentReference[oaicite:2]{index=2}
- The LSTM computation at each time step must be implemented manually:
  - input gate, forget gate, output gate, candidate cell
  - cell state and hidden state updates
- You are allowed to change hyperparameters and **may change the optimizer** to improve accuracy. :contentReference[oaicite:3]{index=3}

## LSTM Formulation (Per Time Step)
The assignment provides the standard LSTM equations:
- \( i_t = \sigma(W_i h_{t-1} + U_i x_t + b_i) \)
- \( f_t = \sigma(W_f h_{t-1} + U_f x_t + b_f) \)
- \( o_t = \sigma(W_o h_{t-1} + U_o x_t + b_o) \)
- \( \tilde{c}_t = \tanh(W_c h_{t-1} + U_c x_t + b_c) \)
- \( c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \)
- \( h_t = o_t \odot \tanh(c_t) \)

After the final time step:
- \( logits = W_{out} h_t + b_{out} \) :contentReference[oaicite:4]{index=4}

## What You Need to Implement

### 1. Manual LSTM Cell
Implement a custom module:
- Define weights/biases for:
  - Forget gate
  - Input gate
  - Output gate
  - Candidate cell
- In `forward()`:
  1. Concatenate `x` and `h_prev`
  2. Compute gates with `sigmoid`
  3. Compute candidate with `tanh`
  4. Update `c_t` and `h_t` :contentReference[oaicite:5]{index=5}

### 2. Manual LSTM Classifier
- Initialize `h_t` and `c_t` to zeros.
- Unroll over 28 time steps.
- Use the **last hidden state** for classification via a fully connected layer. :contentReference[oaicite:6]{index=6}

### 3. Training Loop
- Use `CrossEntropyLoss`.
- Train for multiple epochs.
- Record **training loss**.

### 4. Testing Loop
- Evaluate on **all 10,000 test images**.
- Print **overall test accuracy**. :contentReference[oaicite:7]{index=7}

### 5. Visualization
- Display **10 example test images** with:
  - true label
  - predicted label  
  (Ideally covering digits 0–9 if possible.) :contentReference[oaicite:8]{index=8}

## Grading (Homework — 70% Max)

### Implementation
1. **(30%)** Manual LSTM Cell correctly built from equations  
2. **(15%)** Training + hyperparameter tuning (accuracy-dependent)  
3. **(5%)** Correct testing loop over full test set  
4. **(5%)** Visualization of 10 example predictions  

### Questions
5. **(5%)** Roles of forget/input/output/candidate components  
6. **(5%)** What you tuned and how it affected accuracy  
7. **(5%)** Simple RNN vs LSTM: which is better and when  

(Details follow the provided handout.) :contentReference[oaicite:9]{index=9}

## Hyperparameters
The template provides default hyperparameters (e.g., `hidden_size`, `batch_size`, `learning_rate`, `num_epochs`).  
You are required to **adjust them** to maximize test accuracy. Examples of common tuning directions:
- Increase `hidden_size`
- Adjust `learning_rate`
- Change `batch_size`
- Train longer with more `epochs` :contentReference[oaicite:10]{index=10}

## Dataset
- **MNIST** is loaded via `torchvision.datasets.MNIST`.
- Each image is reshaped to `(batch, 28, 28)` to represent a sequence of length 28. :contentReference[oaicite:11]{index=11}

## Suggested Repository Structure
```text
.
├─ 112101014_Lab7_Homework.ipynb
├─ 112101014_Lab7_Homework.pdf
├─ data/                       # auto-downloaded MNIST
└─ README.md
````

## Environment

Recommended:

* Python 3.x
* torch
* torchvision
* numpy
* matplotlib

Install:

```bash
pip install torch torchvision numpy matplotlib
```

## How to Run

### Jupyter Notebook

Open and run:

```bash
jupyter notebook 112101014_Lab7_Homework.ipynb
```

The notebook will:

1. Download MNIST
2. Build manual LSTM cell + classifier
3. Train the model
4. Evaluate test accuracy
5. Visualize 10 prediction examples

## Report & Submission

Upload to E3:

1. **Report** with screenshots in the last pages:

   * (a) Hyperparameters
   * (b) Training loss record
   * (c) Test accuracy
   * (d) Prediction results
2. **Code** in `.py` or `.ipynb`

Naming:

* `StudentID_Lab7_Homework.pdf`
* `StudentID_Lab7_Homework.py` or `StudentID_Lab7_Homework.ipynb`

Deadline:

* **Sunday 21:00 PM** 

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your implementation and report are your own work. 

# Machine Learning Lab 7 (In-Class) — Vanilla RNN from Scratch (Sentence Sentiment)

This repository contains my **Lab 7 In-Class Assignment** for the Machine Learning laboratory on **LSTM-RNN**.  
In this in-class task, I implement a **basic Vanilla RNN** for simple **sequence (sentence) classification** using **PyTorch**, but **without** any high-level RNN modules or built-in optimizers. The model predicts whether a short sentence is **"good" (1)** or **"bad" (0)**. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Implement a **Vanilla RNN** hidden state update step-by-step over a sequence.
- Build a simple **linear output layer** for sentence classification.
- Perform **manual SGD parameter updates** using gradients.
- Understand how the **hidden state summarizes past information**.
- Visualize the **training loss curve** and test predictions on simple sentences. :contentReference[oaicite:1]{index=1}

---

## Key Rules / Constraints
- **Do not use**:
  - `nn.RNN`
  - `nn.LSTM`
  - `optim.SGD` (or other built-in optimizers)
- Implement the RNN computation and parameter updates manually.
- Use the provided toy vocabulary and training samples. :contentReference[oaicite:2]{index=2}

---

## Model Formulation

### Hidden State Update
For each time step:
\[
s_t = \phi(Ws_{t-1} + Ux_t + b)
\]
where:
- \( s_t \) is the hidden state
- \( x_t \) is a one-hot word vector
- \( \phi \) is an activation function (as used in the template)

### Output Layer
After the final time step:
\[
\text{logits} = W_{\text{out}} s_T + b_{\text{out}}
\]
Prediction is obtained by `argmax(logits)`. :contentReference[oaicite:3]{index=3}

---

## What I Implemented

### 1. Vocabulary & One-Hot Encoding
A small fixed vocab:
```python
vocab = { "The": 0, "movie": 1, "is": 2, "good": 3, "bad": 4 }
````

and `word_to_onehot()`.

### 2. Weight Initialization

Manually initialize:

* `W` (n × n)
* `U` (n × m)
* `b` (n × 1)
* `W_out` (2 × n)
* `b_out` (2 × 1) 

### 3. Forward Pass (RNN Unrolling)

For each sentence:

* start with `s_prev = zeros(n, 1)`
* iterate through words:

  * compute `s_t`
  * update `s_prev`

### 4. Manual SGD Updates

After `loss.backward()`:

* update each parameter with:
  [
  \theta \leftarrow \theta - \eta \cdot \nabla_\theta
  ]
* then zero out gradients. 

### 5. Visualization

Plot the **training loss curve** across epochs.

### 6. Testing

Verify the model predicts:

* `["The", "movie", "is", "bad"]` → 0
* `["The", "movie", "is", "good"]` → 1 

---

## Grading (In-Class — 30% Max)

**Implementation**

1. (5%) Correctly initialize all weights
2. (5%) Correct hidden state update + output logits
3. (5%) Correct manual SGD updates
4. (5%) Correct predictions on test sentences

**Questions**
5. (5%) What does ( s_t ) represent at each time step?
6. (5%) Why are RNNs hard to train? 

---

## Files

```text
.
├─ 112101014_Lab7_InClass.ipynb
├─ 112101014_Lab7_InClass.pdf
└─ README.md
```

---

## Environment

Recommended:

* Python 3.x
* torch
* matplotlib

Install:

```bash
pip install torch matplotlib
```

---

## How to Run

### Jupyter Notebook

```bash
jupyter notebook 112101014_Lab7_InClass.ipynb
```

Run all cells to:

* train the RNN with manual updates
* print loss checkpoints
* show the loss curve
* print test predictions

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab7_InClass.pdf`

   * Include screenshots of your results in the last pages.
2. **Code**: `StudentID_Lab7_InClass.py` or `StudentID_Lab7_InClass.ipynb`

Deadline: **16:20 PM**. 

---

## Academic Integrity

Plagiarism is strictly prohibited. 

# Machine Learning Lab 8 (Homework) — Vision Transformer (ViT) for Defect Detection

This repository contains my implementation for **Machine Learning Laboratory: Transformer Homework**.

In this assignment, I implement a **Vision Transformer (ViT)** from scratch using **PyTorch** and apply it to an **industrial defect detection** dataset with **6 classes**. Each image is treated as a sequence of non-overlapping patches, and classification is performed using a **[CLS] token (or pooling)** followed by an MLP head.

---

## Objectives
- Understand how ViT converts images into patch tokens and models global relationships with self-attention.
- Manually implement:
  - **Patch embedding**
  - **Positional encoding**
  - **Transformer encoder** (Multi-Head Self-Attention, MLP, LayerNorm, residual connections)
  - **Classification head**
- Train the model on a real-world defect dataset.
- Evaluate performance on an unseen test split.
- Tune hyperparameters to improve test accuracy.

---

## Key Rules / Constraints
- **No prebuilt ViT libraries/modules**:
  - Do NOT use `torchvision.models.vit`
  - Do NOT use `timm.create_model`
  - Do NOT use high-level Transformer shortcuts
- The ViT architecture must be implemented **manually**.
- You must **split the raw dataset yourself** into:
  - **70% training**
  - **30% testing**
- Hyperparameter tuning is required.

---

## Dataset
The defect dataset contains 6 categories:

```text
crazing
inclusion
patches
pitted_surface
rolled-in_scale
scratches
````

You will:

1. Upload and unzip the dataset.
2. Split by class into train/test folders.
3. Use `ImageFolder` with custom transforms.

### Suggested transforms

* Resize to **28×28**
* Convert to **grayscale (1 channel)**
* Normalize (e.g., mean=0.5, std=0.5)

---

## Model Overview

### ViT Pipeline

1. **Patch Embedding**

   * Split image into **non-overlapping P×P patches**.
   * Flatten each patch.
   * Project to embedding dimension `dim`.
   * A practical implementation uses `Conv2d` where:

     * `kernel_size = patch_size`
     * `stride = patch_size`

2. **Add [CLS] Token**

   * Prepend a learnable token to the patch sequence.

3. **Positional Embedding**

   * Add learnable positional embeddings to retain spatial order.

4. **Transformer Encoder**

   * Repeated blocks of:

     * `LayerNorm → Multi-Head Self-Attention → Residual`
     * `LayerNorm → FeedForward (MLP) → Residual`

5. **Classification Head**

   * Use the **[CLS] output** (or mean pooling) → MLP → logits.

---

## What I Implemented

* `PreNorm`
* `FeedForward`
* `Attention`
* `Transformer`
* `ViT`
* Training loop
* Evaluation loop
* Visualization tools:

  * **Confusion matrix**
  * **Classwise example predictions** (24 images, 4 per class)

---

## Hyperparameters (Example Starting Point)

You should adjust these for best performance:

```python
image_size = 28
patch_size = 4
num_classes = 6
dim = 64
depth = 6
heads = 4
mlp_dim = 128
dropout = 0.1
emb_dropout = 0.1
lr = 3e-3
batch_size = 32 or 64
epochs = 30–100
```

---

## Grading Overview

* **ViT model design from scratch** (major portion)
* **Training & tuning** (must reach **≥ 60%** test accuracy; higher is better)
* Correct **evaluation loop**
* Required **visualizations**
* Report quality and conceptual answers

---

## Required Report Items

Include screenshots of:

1. **Model summary** and **total parameters**
2. **Final training result** (epochs, final test loss & accuracy)
3. **Confusion matrix**
4. **24 example predictions** (4 images per class)

Also answer questions on:

* Patch embedding & positional encoding
* Hyperparameter tuning choices
* ViT vs CNN comparison
* Final accuracy discussion
* ViT end-to-end explanation based on the original paper

---

## Repository Structure (Suggested)

```text
.
├─ 112101014_Lab8_Homework.ipynb
├─ 112101014_Lab8_Homework.pdf
├─ dataset.zip
├─ dataset/                       # unzipped raw data
├─ dataset_split/
│  ├─ train/
│  └─ test/
├─ 112101014_Lab8_Homework.pth    # trained weights
└─ README.md
```

---

## Environment

Recommended:

* Python 3.x
* torch
* torchvision
* einops
* numpy
* matplotlib
* seaborn
* scikit-learn (for confusion matrix)

Install:

```bash
pip install torch torchvision einops numpy matplotlib seaborn scikit-learn
```

---

## How to Run

### 1) Prepare data

* Put your dataset zip in the project root.
* Update:

  ```python
  zip_path = "dataset.zip"
  ```
* Run the unzip + split cells to create:

  ```text
  dataset_split/train
  dataset_split/test
  ```

### 2) Train

* Initialize the ViT model.
* Set optimizer (e.g., Adam/AdamW).
* Increase `N_EPOCHS` beyond the template default.

### 3) Evaluate & Visualize

* Run the evaluation function to report:

  * test loss
  * test accuracy
* Generate:

  * confusion matrix
  * 24 classwise prediction examples

### 4) Save model

```python
torch.save(model.state_dict(), "112101014_Lab8_Homework.pth")
```

---

## Submission Checklist

Upload **all three files** to E3:

1. `StudentID_Lab8_Homework.pdf`
2. `StudentID_Lab8_Homework.py` or `.ipynb`
3. `StudentID_Lab8_Homework.pth`

Missing any one file may result in **no grade**.

---

## Academic Integrity

Plagiarism is strictly prohibited. Ensure your implementation and report are your own work.

```

Based on the provided Lab 8 Transformer Homework handout. :contentReference[oaicite:0]{index=0}
```

# Machine Learning Lab 8 (In-Class) — GAN on MNIST (PyTorch)

This repository contains my **Lab 8 In-Class Assignment** for the Machine Learning laboratory on **Generative Adversarial Networks (GANs)**.  
The goal is to implement and complete a GAN training loop in **PyTorch** to generate handwritten **MNIST** digit images, focusing on the **last digit of my student ID** as the target style. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Understand the training dynamics of a **GAN**.
- Implement the **adversarial loss** using **binary cross-entropy (BCE)**.
- Train:
  - a **Generator (G)** that maps random noise to digit-like images
  - a **Discriminator (D)** that distinguishes real vs fake images
- Visualize generated outputs across epochs to observe model evolution. :contentReference[oaicite:1]{index=1}

---

## Key Rules / Constraints
- Use **PyTorch** for model building and training.
- **Do not use higher-level GAN wrappers/libraries**.
- Train the GAN **only on a single digit class**:
  - If your student ID ends in `k`, set `target_digit = k`. :contentReference[oaicite:2]{index=2}

---

## Dataset
- **MNIST (train split)** is downloaded automatically.
- The dataset is **filtered** to only include the `target_digit`.
- The training set is **limited to 5000 samples** for this in-class task. :contentReference[oaicite:3]{index=3}

---

## Model Architecture (My Implementation)
This in-class submission uses a simple **MLP-based GAN**:

### Generator
- Input: `z_dim = 100` Gaussian noise
- MLP with LeakyReLU + BatchNorm stacks
- Output: 784-dim vector reshaped to 28×28
- Final activation: **Tanh** to match normalized image range `[-1, 1]`. :contentReference[oaicite:4]{index=4}

### Discriminator
- Input: flattened 784-dim image
- MLP with LeakyReLU
- Final activation: **Sigmoid** for real/fake probability. :contentReference[oaicite:5]{index=5}

---

## Training Setup
- Loss: `nn.BCELoss()`
- Optimizers: **Adam**
  - learning rate `lr = 0.0002`
  - `betas = (0.5, 0.999)`
- Batch size: `64`
- Epochs: `100`
- Device: CUDA if available. :contentReference[oaicite:6]{index=6}

Training procedure:
1. Sample real images (target digit only).
2. Generate fake images from random noise.
3. Train **D** on:
   - real labeled as 1
   - fake labeled as 0
4. Train **G** to fool **D**:
   - fake labeled as 1 for generator loss. 

---

## Outputs & Visualization
The code saves results under:
```text
gan_results/
````

Including:

* Real digit grid before training
* Generated grids at:

  * Epoch 1, 10, 20, ..., 100
* Loss curve plot for D and G
* Final real vs generated comparison. 

---

## Grading (In-Class — 30% Max)

* (10%) Implement GAN training loop (G/D + BCE)
* (10%) Code runs end-to-end and shows losses + generated samples
* (5%) Correct target digit style (last digit of student ID)
* (5%) Brief discussion of results
  Deadline: **16:20 PM**. 

---

## Repository Structure

```text
.
├─ 112101014_Lab8_InClass.py
├─ 112101014_Lab8_InClass.pdf
├─ gan_results/
│  ├─ real_digit_<k>.png
│  ├─ generated_epoch_*.png
│  ├─ loss_plot.png
│  └─ final_comparison.png
└─ README.md
```

---

## Environment

Recommended:

* Python 3.x
* torch
* torchvision
* numpy
* matplotlib

Install:

```bash
pip install torch torchvision numpy matplotlib
```

---

## How to Run

```bash
python 112101014_Lab8_InClass.py
```

Before running, ensure:

```python
target_digit = <last_digit_of_student_id>
```

Then the script will:

1. Download and filter MNIST
2. Train the GAN
3. Save images and loss plots to `gan_results/`. 

---

## Academic Integrity

Plagiarism is strictly prohibited. 

# Machine Learning Lab 9 (Homework) — GAN vs CycleGAN on FashionMNIST & CIFAR-10

This repository contains my implementation for **Machine Learning Laboratory: GAN Homework**.

The homework extends the in-class GAN assignment to **new datasets** and introduces a **CycleGAN-style** approach, with an emphasis on **multi-class generation**, **style mimicry**, and a structured **comparison between GAN and CycleGAN**. :contentReference[oaicite:0]{index=0}

---

## Objectives
- Reuse and adapt in-class GAN code to:
  - **FashionMNIST**
  - **CIFAR-10**
- Train generative models on **at least 3 classes per dataset**.
- Implement a **CycleGAN-style generator and discriminator** for the same class sets.
- Explore:
  - diversity vs mode collapse
  - class structure preservation
  - style mimic behavior
- Provide clear visual and/or simple quantitative comparisons. :contentReference[oaicite:1]{index=1}

---

## Assignment Overview

### Part 1 — GAN on New Datasets
For both FashionMNIST and CIFAR-10:
- Pick **≥ 3 classes** per dataset.
- Train a GAN **per class** (via filtering / class-conditional-by-selection).
- Visualize:
  - **Real**
  - **Fake**
  - **Mimic** (style-targeted generation)

Implementation options:
- One model per class, or
- A unified loader that trains class-by-class. :contentReference[oaicite:2]{index=2}

### Part 2 — CycleGAN for Same Tasks
- Adapt a basic CycleGAN-style architecture.
- Train CycleGAN on the **same chosen classes**.
- Visualize:
  - **Real**
  - **Fake**
  - **Mimic**
- Optionally experiment with:
  - one-directional mapping
  - cycle consistency variations. :contentReference[oaicite:3]{index=3}

### Part 3 — GAN vs CycleGAN Comparison
- Choose at least one comparison angle:
  - visual diversity
  - sharpness
  - mimic accuracy / style fidelity
  - training stability
- Provide:
  - **≥ 3 side-by-side comparisons** for the same class across both models. :contentReference[oaicite:4]{index=4}

---

## Mimic Mode (Style Targeting)
This homework requires an explicit **mimic mechanism**:

- **GAN**: optimize the **latent vector** \( z \) to match a target real image style.
- **CycleGAN**: optimize the **input image** (or mapping behavior) to mimic a target real instance.

You should generate **multiple mimic samples** demonstrating convergence toward the target style. :contentReference[oaicite:5]{index=5}

---

## Grading (Homework — 70% Max)

### Implementation (50%)
1. **(15%)** GAN on FashionMNIST & CIFAR-10  
   - ≥ 3 classes per dataset  
   - real/fake/mimic visualizations  
2. **(15%)** CycleGAN on the same classes  
   - generator/discriminator + cycle consistency  
   - outputs visualized  
3. **(10%)** Mimic mode implementation  
4. **(10%)** GAN vs CycleGAN comparison  
   - ≥ 3 visual comparisons  

### Questions (20%)
- Which model is more realistic/varied and why?
- How did mimic mode perform across models?
- How would you improve results?  
  (architecture, losses, normalization, training tricks, etc.) :contentReference[oaicite:6]{index=6}

---

## Datasets
- **FashionMNIST** (grayscale, 28×28)
- **CIFAR-10** (RGB, 32×32)

Example class choices:
- FashionMNIST: T-shirt/top (0), Trouser (1), Coat (4), etc.
- CIFAR-10: Airplane (0), Cat (3), Dog (5), etc. :contentReference[oaicite:7]{index=7}

---

## Suggested Repository Structure
```text
.
├─ 112101014_Lab9_Homework.ipynb
├─ 112101014_Lab9_Homework.pdf
├─ src/                          # (optional) modularized code
│  ├─ gan.py
│  ├─ cyclegan.py
│  ├─ mimic.py
│  ├─ data.py
│  └─ utils.py
├─ outputs/
│  ├─ fashion/
│  │  ├─ class_0/
│  │  ├─ class_1/
│  │  └─ class_4/
│  └─ cifar/
│     ├─ class_0/
│     ├─ class_3/
│     └─ class_5/
└─ README.md
````

---

## Environment

Recommended:

* Python 3.x
* torch
* torchvision
* numpy
* matplotlib

Install:

```bash
pip install torch torchvision numpy matplotlib
```

---

## How to Run

### Notebook

```bash
jupyter notebook 112101014_Lab9_Homework.ipynb
```

### (Optional) Script Mode

If you refactor into scripts:

```bash
python src/gan.py --dataset fashion --classes 0 1 4
python src/gan.py --dataset cifar --classes 0 3 5

python src/cyclegan.py --dataset fashion --classes 0 1 4
python src/cyclegan.py --dataset cifar --classes 0 3 5

python src/mimic.py --model gan --dataset fashion --class_id 0
python src/mimic.py --model cyclegan --dataset cifar --class_id 3
```

---

## Expected Outputs

For each chosen class:

* **Real / Fake / Mimic** image grids
* Training logs for G/D (loss trends)
* At least **3 GAN vs CycleGAN side-by-side comparisons**

Include screenshots of these results in the last pages of the provided PDF report. 

---

## Submission

Upload to E3:

1. **Report**: `StudentID_Lab9_Homework.pdf`
2. **Code**: `StudentID_Lab9_Homework.py` or `.ipynb`

Deadline:

* **Sunday, 21:00 PM**

Plagiarism is strictly prohibited. 

Based on the provided **Machine Learning Final Project** handout. 

# Machine Learning Final Project 2025 — Metallic Surface Defect Detection

This repository contains my implementation for the **Machine Learning Final Project** on **industrial inspection with object detection**.

The task is to train and evaluate an object detection model on a **custom metallic surface defect dataset** with **6 defect classes**, then submit predictions to a **private Kaggle leaderboard** using a hidden test set.

---

## Objectives
- Apply deep learning object detection techniques to an industrial inspection problem.
- Train and evaluate an improved detection model.
- Generate a `submission.csv` for Kaggle evaluation.

---

## Dataset
The dataset contains metallic surface images with **6 defect classes**:
1. Crazing  
2. Inclusion  
3. Patches  
4. Pitted Surface  
5. Rolled-in Scale  
6. Scratches  

**Data split provided**
- **Training set:** 1,440 images with bounding box annotations  
- **Test set:** 360 unlabeled images (used for Kaggle submission)

You may further split the training set into train/validation as needed.

---

## Competition Rules (Important)
- **Individual work only.** No teamwork, code sharing, or joint submissions.
- **You may use pretrained models, but must modify or improve the architecture.**
  - Do **not** use a pretrained model as-is.
  - Allowed improvements: add layers, adjust detection heads, fine-tune, or other task-relevant changes.
- **Submission limit:** max 20 submissions per day on Kaggle.
- **Evaluation metric:** **IoU @ 0.5** (higher is better).
- **Baseline requirement:** your model must achieve at least **70% mAP @ IoU 0.5**.
- **Final Kaggle deadline:** **June 26, 23:59**.

---

## Kaggle Registration Notes
- Join the competition using the invitation link provided in the handout.
- Your Kaggle email must match your E3 email.
- Change your Kaggle display name to:
  - `StudentID_Name` (e.g., `112101014_YourName`)

---

## Method Overview
I trained an object detection model with the following strategy:

- **Backbone:** pretrained (e.g., ResNet / CSP / equivalent)
- **Architecture modifications (required):**
  - Customized detection head
  - Adjusted feature pyramid / neck
  - Tuned anchor settings (if applicable)
  - Class-specific loss/augmentation refinements
- **Training improvements:**
  - Stronger augmentations (random resize/crop, color jitter, blur/noise)
  - Better learning rate scheduling
  - Longer training with early stopping based on validation mAP

> The exact design and ablation results are documented in the report.

---

## Repository Structure (Suggested)
```text
.
├─ notebooks/
│  └─ 0528_2.ipynb
├─ src/
│  ├─ train.py
│  ├─ infer.py
│  ├─ dataset.py
│  ├─ transforms.py
│  └─ utils.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  └─ predictions/
├─ submission.csv
└─ README.md
````

---

## Environment

Recommended:

* Python 3.9+
* PyTorch
* torchvision
* numpy
* pandas
* opencv-python
* matplotlib

Install:

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib
```

---

## Training

### Option A — Train on Kaggle

You can train directly in a Kaggle Notebook without downloading data locally.

### Option B — Train Locally

1. Place the dataset in a local folder:

```text
data/
├─ train/
│  ├─ images/
│  └─ annotations/
└─ test/
    └─ images/
```

2. Run training:

```bash
python src/train.py \
  --data_dir data \
  --epochs 50 \
  --batch_size 8 \
  --img_size 640 \
  --lr 1e-3 \
  --save_dir outputs/checkpoints
```

---

## Validation

During training, evaluate using:

* **mAP @ IoU 0.5**
* Per-class AP
* Loss curves

---

## Inference & Submission

1. Generate predictions on the test set:

```bash
python src/infer.py \
  --checkpoint outputs/checkpoints/best.pth \
  --test_dir data/test/images \
  --out_dir outputs/predictions
```

2. Build `submission.csv` following the competition format:

```bash
python src/utils.py --build_submission \
  --pred_dir outputs/predictions \
  --output submission.csv
```

3. Upload `submission.csv` to Kaggle.

---

## Tips for Better Performance

* Verify annotation format and class mapping carefully.
* Start from a strong baseline detector, then **document your architectural changes**.
* Use a stable train/val split.
* Monitor overfitting; tune augmentation and weight decay.
* Try:

  * higher-resolution training
  * balanced sampling if classes are imbalanced
  * class-aware loss weights

---

## Academic Integrity

This is an **individual project**.
Any code sharing or collaboration that violates the rules may lead to severe penalties.

---

## Acknowledgments

* Course: Machine Learning (NYCU)
* Instructor: Prof. Hsien-I Lin
* TAs: Satrio Sanjaya, Muhammad Ahsan

