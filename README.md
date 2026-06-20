# Financial Markets ML Thesis

This repository contains the codebase developed for my MSc thesis in Business Analytics & Data Science. The project focuses on the development of machine learning pipelines for financial market signal classification across multiple asset classes, including cryptocurrencies, forex, stocks, and indices.

The objective of the project is to build an end-to-end analytical workflow that transforms raw financial time-series data into supervised machine learning datasets, engineers technical and time-based features, trains classification models, and evaluates their predictive performance under realistic temporal validation settings.

## Project Scope

The project covers the following main stages:

* Data preprocessing and cleaning
* Technical indicator transformation
* Feature engineering for short-term and long-horizon market behaviour
* Target engineering for classification-based trading outcomes
* Chronological train-validation-test splitting
* Class imbalance handling
* Machine learning model training
* Model evaluation using classification and performance-oriented metrics
* Visualization of model results

## Asset Classes

The broader thesis framework is designed around four financial market categories:

* Cryptocurrencies
* Forex pairs
* Stocks
* Market indices

Each asset class follows a similar modelling logic, while allowing for market-specific preprocessing and evaluation where necessary.

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* joblib
* Jupyter Notebook
* VS Code

## Repository Structure

```text
financial-markets-ml-thesis/
│
├── src/                    # Core Python scripts
├── config/                 # Example configuration files
├── data/                   # Data instructions only; raw datasets are not included
├── notebooks/              # Exploratory or demonstration notebooks
├── reports/figures/        # Selected non-sensitive figures
├── docs/                   # Methodology notes
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from version control
└── README.md               # Project documentation
```

## Data Availability

The original datasets are not included in this public repository. This is intentional, due to file size, licensing, and reproducibility considerations. The `data/` directory contains instructions about the expected local data structure.

## Status

This repository is currently under development as part of an MSc thesis project. The codebase will be gradually cleaned, documented, and expanded as the thesis implementation progresses.

## Author

Gregory Emmanouilidis
MSc Business Analytics & Data Science
Data & Business Analytics / SAP Digital Transformation
