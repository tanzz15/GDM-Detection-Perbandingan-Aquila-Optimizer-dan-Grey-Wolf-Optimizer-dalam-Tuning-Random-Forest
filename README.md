# GDM Detection Using Random Forest, Aquila Optimizer, and Grey Wolf Optimizer

A Machine Learning project to detect Gestational Diabetes Mellitus (GDM) using Random Forest and compare two metaheuristic optimization algorithms: **Aquila Optimizer (AO)** and **Grey Wolf Optimizer (GWO)** for hyperparameter tuning.

## Project Overview

Gestational Diabetes Mellitus (GDM) is a type of diabetes that develops during pregnancy. This project explores the application of machine learning to detect potential GDM cases based on demographic and clinical features.

The main objective is to compare the performance of three Random Forest models:

- **Random Forest Baseline:** Model without hyperparameter optimization.
- **Random Forest + GWO:** Model optimized using Grey Wolf Optimizer.
- **Random Forest + AO:** Model optimized using Aquila Optimizer.

## Dataset

The dataset consists of **1,372 records** with 29 selected predictor features and a binary target class.

- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/k5c8rfgsn9/1)
- **DOI:** [10.17632/k5c8rfgsn9.1](https://doi.org/10.17632/k5c8rfgsn9.1)
- **Target Classes:** GDM Positive and GDM Negative
- **Selected Predictors:** 29 features
- **Final Modeling Features:** 30 features, including BMI

### Feature Engineering

An additional feature, Body Mass Index (BMI), is generated from weight and height.

\[
BMI = \frac{Weight}{(Height/100)^2}
\]

The resulting dataset contains 30 features for model training.

## Methodology

The project workflow consists of:

1. Data preprocessing and cleaning.
2. Feature selection.
3. BMI feature engineering.
4. Stratified train-test split (80:20).
5. Random Forest model development.
6. Hyperparameter tuning using AO and GWO.
7. Model evaluation and comparison.

### Optimization Algorithms

**Grey Wolf Optimizer (GWO)**

A metaheuristic optimization algorithm inspired by the social hierarchy and hunting behavior of grey wolves.

**Aquila Optimizer (AO)**

A metaheuristic optimization algorithm inspired by the hunting strategies of the Aquila eagle.

Both algorithms are used to search for optimal Random Forest hyperparameters.

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Computation Time

The evaluation compares the baseline Random Forest model with the two optimized models.

## Application Demo

The trained models are integrated into a web application built with **Gradio**.

Users can input demographic and clinical features and compare prediction results from all three models:

- Random Forest Baseline
- Random Forest + GWO
- Random Forest + AO

**Live Demo:** [GDM Detection Web Application](https://huggingface.co/spaces/tanzz15/gdm-ao-gwo-rf)

## Repository Structure

```text
├── artifacts/
├── GDM.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Aquila Optimizer
- Grey Wolf Optimizer
- Gradio
- Joblib

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/tanzz15/GDM-Detection-Perbandingan-Aquila-Optimizer-dan-Grey-Wolf-Optimizer-dalam-Tuning-Random-Forest.git
```

Navigate to the project directory:

```bash
cd GDM-Detection-Perbandingan-Aquila-Optimizer-dan-Grey-Wolf-Optimizer-dalam-Tuning-Random-Forest
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open the local URL displayed in the terminal.

## Research Objective

This project investigates the effectiveness of Aquila Optimizer and Grey Wolf Optimizer in tuning Random Forest hyperparameters for Gestational Diabetes Mellitus detection.

The comparison aims to identify which optimization algorithm provides better overall model performance.

## Disclaimer

This project is developed for academic and research purposes. The predictions are not intended to replace professional medical diagnosis or clinical judgment.
