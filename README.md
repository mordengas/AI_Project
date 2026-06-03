# AI_Project

## 🧠 Project Overview
A simple university project focused on predicting loan approvals using machine learning. It features a neural network built in MATLAB for data processing and classification.

## ⚙️ Features and Workflow

### 1. Data Preprocessing (MATLAB)
The `projekt.m` script prepares the raw data for neural network training:
* **Handling Missing Data:** If the missing data ratio in a column exceeds 5%, missing numerical values are imputed using the median, and categorical values using the mode. Otherwise, rows with missing values are removed.
* **Label Encoding:** Categorical variables such as `Gender`, `Married`, `Education`, `Self_Employed`, and `Property_Area` are converted into numeric values representing their respective classes.
* **Normalization:** Financial features (e.g., `ApplicantIncome`, `LoanAmount`) are scaled using Min-Max normalization to a range of <0, 1>.

### 2. Artificial Intelligence Model (MATLAB)
The main prediction engine is a Feed-Forward Neural Network:
* **Architecture:** Built using `newff` with 30 neurons in the hidden layer.
* **Activation Functions:** Logistic sigmoid (`logsig`) and linear (`purelin`).
* **Data Split:** The dataset is divided into 80% training, 10% validation, and 10% testing sets.
* **Evaluation:** The network's performance is evaluated using a Confusion Matrix, and key metrics are calculated: *Accuracy*, *Precision*, and *Recall*.

## 📂 File Structure
* `projekt.m` - The main executable script written in MATLAB.
* `py.py` - A helper exploratory script written in Python.
* `train.csv` / `train_u6lujuX_CVtuZ9i.csv` - Required dataset files (not included in the repository, must be downloaded/provided).

## 🛠️ System Requirements
* **MATLAB** (requires the *Deep Learning Toolbox* for building the neural network)
* **Python 3.x**
  * `pandas` library

## 🚀 How to Run
1. Place your `train.csv` dataset in the root directory of the project.
2. (Optional) Run the Python script in your terminal to check data quality: `python py.py`.
3. Open `projekt.m` in MATLAB and run the script to preprocess the data, train the model, and view the evaluation metrics.
