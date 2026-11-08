# GDM Detection Using Random Forest, Aquila Optimizer, and Grey Wolf Optimizer

<p align="center">
  A Machine Learning project for Gestational Diabetes Mellitus (GDM) detection using Random Forest, Aquila Optimizer (AO), and Grey Wolf Optimizer (GWO).
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/tanzz15/gdm-ao-gwo-rf">Live Demo</a> •
  <a href="https://github.com/tanzz15">GitHub Profile</a>
</p>

---

## About the Project

This project focuses on detecting Gestational Diabetes Mellitus (GDM) using Machine Learning techniques. It combines the Random Forest classification algorithm with two metaheuristic optimization algorithms: Aquila Optimizer (AO) and Grey Wolf Optimizer (GWO).

The project uses a dataset containing **1,372 records** with 29 selected predictor features. An additional feature, Body Mass Index (BMI), is generated through feature engineering, resulting in 30 features for model training.

The main objective is to compare the performance of Random Forest without optimization against Random Forest optimized using AO and GWO.

## Technologies & Tools

- **Python** – Main programming language.
- **Pandas** – Data manipulation and preprocessing.
- **NumPy** – Numerical computation.
- **Scikit-learn** – Machine Learning model development and evaluation.
- **Joblib** – Model serialization and loading.
- **Random Forest** – Classification algorithm.
- **Aquila Optimizer (AO)** – Hyperparameter optimization.
- **Grey Wolf Optimizer (GWO)** – Hyperparameter optimization.
- **MLflow** – Experiment tracking.
- **Gradio** – Web application development.
- **Hugging Face Spaces** – Application deployment.

## Project Workflow

The project follows these main steps:

1. **Data Collection** – Using the Gestational Diabetes Mellitus dataset.
2. **Data Preprocessing** – Cleaning and preparing the dataset.
3. **Feature Selection** – Selecting 29 predictor features.
4. **Feature Engineering** – Calculating BMI and adding it as an additional feature.
5. **Data Splitting** – Dividing the dataset into training and testing sets using an 80:20 stratified split.
6. **Model Development** – Building a Random Forest classification model.
7. **Hyperparameter Optimization** – Using AO and GWO to search for optimal Random Forest parameters.
8. **Model Evaluation** – Comparing the three models using classification metrics.
9. **Deployment** – Integrating the trained models into a web application using Gradio and Hugging Face Spaces.

## Machine Learning Methods

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce classification predictions.

### Aquila Optimizer (AO)

Aquila Optimizer is a nature-inspired metaheuristic algorithm based on the hunting strategies of Aquila eagles. In this project, AO is used to optimize Random Forest hyperparameters.

### Grey Wolf Optimizer (GWO)

Grey Wolf Optimizer is a nature-inspired optimization algorithm based on the social hierarchy and hunting behavior of grey wolves. In this project, GWO is used to optimize Random Forest hyperparameters.

## Model Comparison

This project evaluates three different models:

| Model | Description |
|---|---|
| Random Forest Baseline | Random Forest without optimization |
| Random Forest + AO | Random Forest optimized using Aquila Optimizer |
| Random Forest + GWO | Random Forest optimized using Grey Wolf Optimizer |

## Evaluation Metrics

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Computation Time

These metrics are used to compare classification performance and computational efficiency across the three models.

## Dataset Information

- **Dataset:** Gestational Diabetes Mellitus (GDM)
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/k5c8rfgsn9/1)
- **DOI:** [10.17632/k5c8rfgsn9.1](https://doi.org/10.17632/k5c8rfgsn9.1)
- **Total Records:** 1,372
- **Selected Predictor Features:** 29
- **Additional Feature:** BMI
- **Final Modeling Features:** 30
- **Target:** GDM Positive / GDM Negative

### BMI Calculation

BMI is calculated using weight and height:

\[
BMI = \frac{Weight}{(Height/100)^2}
\]

## Live Demo

Try the deployed GDM detection application:

**https://huggingface.co/spaces/tanzz15/gdm-ao-gwo-rf**

The application provides prediction results from three models: Random Forest Baseline, Random Forest + AO, and Random Forest + GWO.

## Developer

**GitHub:** [@tanzz15](https://github.com/tanzz15)

This project demonstrates experience in data preprocessing, feature engineering, Machine Learning development, metaheuristic optimization, model evaluation, experiment tracking, and application deployment.

---

<p align="center">
  Developed as an academic research project.
</p>
