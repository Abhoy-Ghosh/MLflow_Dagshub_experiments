# 🚀 MLflow + DagsHub Experiment Tracking

A beginner-friendly MLOps project demonstrating:

* MLflow Experiment Tracking
* DagsHub Remote Tracking
* Hyperparameter Logging
* Artifact Logging
* Model Logging
* ElasticNet Regression
* Experiment Visualization

Built using:

* Python
* Scikit-learn
* MLflow
* DagsHub

---

# 📌 Project Goal

This project trains an **ElasticNet Regression Model** on the Wine Quality dataset while tracking:

✅ Hyperparameters
✅ Metrics
✅ Artifacts
✅ Trained Models
✅ Experiment Runs

using:

* [MLflow](https://mlflow.org?utm_source=chatgpt.com)
* [DagsHub](https://dagshub.com?utm_source=chatgpt.com)

---

# 🧠 Architecture

```txt id="a1"
Python Script
      ↓
Train ElasticNet Model
      ↓
Track Experiments Using MLflow
      ↓
Store Runs On DagsHub
      ↓
Visualize Metrics & Parameters
      ↓
Register & Save Models
```

---

# 📂 Project Structure

```txt id="a2"
MLflow_Dagshub_experiments/
│
├── main.py
├── metrics.txt
├── mlruns/
├── requirements.txt
├── README.md
└── .venv/
```

---

# ⚙️ Technologies Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Programming Language  |
| Scikit-learn | Machine Learning      |
| MLflow       | Experiment Tracking   |
| DagsHub      | Remote MLflow Hosting |
| Pandas       | Data Processing       |
| NumPy        | Numerical Operations  |

---

# 📦 Installation

## Clone Repository

```bash id="a3"
git clone YOUR_REPOSITORY_URL
```

Enter project folder:

```bash id="a4"
cd MLflow_Dagshub_experiments
```

---

# 🛠 Create Virtual Environment

## Windows

```bash id="a5"
python -m venv .venv
```

Activate environment:

```bash id="a6"
.venv\Scripts\activate
```

---

# 📥 Install Dependencies

```bash id="a7"
pip install -r requirements.txt
```

OR manually:

```bash id="a8"
pip install mlflow dagshub scikit-learn pandas numpy
```

---

# ☁️ DagsHub Setup

## 1️⃣ Create DagsHub Account

Visit:

[DagsHub](https://dagshub.com?utm_source=chatgpt.com)

Login using GitHub.

---

# 🔑 Generate DagsHub Token

Open:

[DagsHub Token Settings](https://dagshub.com/user/settings/tokens?utm_source=chatgpt.com)

Click:

```txt id="a9"
Generate Token
```

Copy the generated token.

---

# 🔐 Authenticate Using Terminal

```bash id="a10"
dagshub login
```

Paste your token.

---

# ▶️ Running The Project

## Default Run

```bash id="a11"
python main.py
```

---

# 🎛 Run With Hyperparameters

```bash id="a12"
python main.py 0.3 0.3
```

Meaning:

| Parameter | Value |
| --------- | ----- |
| alpha     | 0.3   |
| l1_ratio  | 0.3   |

---

# 📊 Dataset

Dataset Used:

```txt id="a13"
Wine Quality Dataset
```

Dataset Source:

[Wine Quality CSV](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv?utm_source=chatgpt.com)

---

# 🤖 Machine Learning Model

Model Used:

```txt id="a14"
ElasticNet Regression
```

Hyperparameters:

* alpha
* l1_ratio

---

# 🧠 Important MLflow & DagsHub Functions

---

## 🔷 `dagshub.init()`

```python id="a15"
dagshub.init(
    repo_owner='Abhoy-Ghosh',
    repo_name='MLflow_Dagshub_experiments',
    mlflow=True
)
```

### ✅ Purpose

Connects local MLflow experiments to DagsHub cloud.

### ✅ What It Does

* Configures tracking URI
* Authenticates automatically
* Enables remote experiment tracking
* Syncs runs to DagsHub

### ✅ Simple Meaning

```txt id="a16"
"Connect this MLflow project to DagsHub"
```

---

## 🔷 `mlflow.set_experiment()`

```python id="a17"
mlflow.set_experiment("Wine Quality Prediction")
```

### ✅ Purpose

Creates/selects an experiment.

### ✅ Simple Meaning

```txt id="a18"
"Group all runs under this experiment"
```

---

## 🔷 `mlflow.start_run()`

```python id="a19"
with mlflow.start_run():
```

### ✅ Purpose

Starts tracking one ML experiment run.

### ✅ Tracks

* parameters
* metrics
* artifacts
* models

### ✅ Simple Meaning

```txt id="a20"
"Start recording this experiment"
```

---

## 🔷 `mlflow.end_run()`

```python id="a21"
mlflow.end_run()
```

### ✅ Purpose

Ends active MLflow run.

### ✅ Useful For

Avoiding:

```txt id="a22"
Run already active
```

errors.

---

## 🔷 `mlflow.log_param()`

```python id="a23"
mlflow.log_param("alpha", alpha)
```

### ✅ Purpose

Logs model configuration values.

### ✅ Examples

* alpha
* epochs
* learning rate
* batch size

### ✅ Simple Meaning

```txt id="a24"
"What settings did this model use?"
```

---

## 🔷 `mlflow.log_metric()`

```python id="a25"
mlflow.log_metric("rmse", rmse)
```

### ✅ Purpose

Logs model performance results.

### ✅ Examples

* accuracy
* r2
* rmse
* mae

### ✅ Simple Meaning

```txt id="a26"
"How good is the model?"
```

---

## 🔷 `mlflow.log_artifact()`

```python id="a27"
mlflow.log_artifact("metrics.txt")
```

### ✅ Purpose

Uploads files to MLflow/DagsHub.

### ✅ Examples

* txt files
* plots
* CSV files
* reports

### ✅ Simple Meaning

```txt id="a28"
"Save this file with the experiment"
```

---

## 🔷 `mlflow.sklearn.log_model()`

```python id="a29"
mlflow.sklearn.log_model(model, "model")
```

### ✅ Purpose

Stores trained machine learning model.

### ✅ Saves

* model weights
* metadata
* environment info

### ✅ Simple Meaning

```txt id="a30"
"Save this trained model"
```

---

## 🔷 `infer_signature()`

```python id="a31"
signature = infer_signature(X_train, predictions)
```

### ✅ Purpose

Captures model input/output schema.

### ✅ Useful For

* deployment
* APIs
* model serving

### ✅ Simple Meaning

```txt id="a32"
"Understand model input and output format"
```

---

# 📈 Metrics Used

| Metric | Meaning                 | Better? |
| ------ | ----------------------- | ------- |
| r2     | Model fit quality       | HIGH    |
| mae    | Mean Absolute Error     | LOW     |
| rmse   | Root Mean Squared Error | LOW     |

---

# 🎛 Hyperparameters Used

| Parameter | Meaning                         |
| --------- | ------------------------------- |
| alpha     | Regularization strength         |
| l1_ratio  | L1 vs L2 regularization balance |

---

# 📊 MLflow Visualizations

MLflow provides:

* experiment comparison
* metric visualization
* hyperparameter analysis
* parallel coordinates plots

Goal:

```txt id="a33"
HIGH r2
LOW mae
LOW rmse
```

---

# 📁 Artifacts Logged

This project logs:

```txt id="a34"
metrics.txt
trained model
experiment metadata
```

---

# 🔥 Complete Workflow

```txt id="a35"
Load Dataset
      ↓
Split Train/Test
      ↓
Start MLflow Run
      ↓
Train ElasticNet Model
      ↓
Calculate Metrics
      ↓
Log Parameters
      ↓
Log Metrics
      ↓
Log Artifacts
      ↓
Save Model
      ↓
Upload To DagsHub
      ↓
Visualize Experiments
```

---

# 🛠 Useful Commands

## Activate Environment

```bash id="a36"
.venv\Scripts\activate
```

---

## Run Training

```bash id="a37"
python main.py 0.5 0.5
```

---

## Start MLflow UI

```bash id="a38"
mlflow ui
```

Open:

```txt id="a39"
http://127.0.0.1:5000
```

---

## Install Dependencies

```bash id="a40"
pip install -r requirements.txt
```

---

## Push To GitHub

```bash id="a41"
git add .
git commit -m "Updated MLflow experiments"
git push
```

---

# 🎯 Learning Outcomes

This project demonstrates:

✅ MLflow Experiment Tracking
✅ DagsHub Remote Tracking
✅ Hyperparameter Tuning
✅ Artifact Management
✅ Model Logging
✅ Experiment Visualization
✅ Basic MLOps Workflow

---

# 🚀 Future Improvements

* FastAPI deployment
* Docker integration
* Automated tuning
* CI/CD pipelines
* Model serving APIs
* Model monitoring

---

# 👨‍💻 Author

Abhoy Ghosh

GitHub:
[GitHub Profile](https://github.com/Abhoy-Ghosh?utm_source=chatgpt.com)

