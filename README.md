# KenyaSMS Shield 🇰🇪

### An Explainable Machine Learning System for SMS Scam Detection and Risk Classification in Kenya

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://kenya-sms-shield.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?logo=streamlit)](https://streamlit.io/)

**KenyaSMS Shield** is a machine learning-powered SMS security system designed to identify potentially fraudulent text messages and classify them according to their level of risk.

The project focuses on SMS scams relevant to the Kenyan context, including messages involving M-Pesa, fraudulent financial transactions, impersonation, suspicious links, requests for sensitive information, and social engineering.

It combines Natural Language Processing (NLP), traditional machine learning, feature engineering, and explainable AI to provide a structured approach to SMS scam detection.

**Try the live application:** [KenyaSMS Shield](https://kenya-sms-shield.streamlit.app/)

---

## Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Risk Classification](#risk-classification)
- [Dataset and Data Preparation](#dataset-and-data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Explainable AI](#explainable-ai)
- [Application Interface](#application-interface)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Running the API](#running-the-api)
- [Running the Notebooks](#running-the-notebooks)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## Live Demo

KenyaSMS Shield is deployed and publicly accessible through **Streamlit Community Cloud**.

<https://kenya-sms-shield.streamlit.app/>

You can access the application directly from your browser without installing Python, downloading the project, or configuring a local development environment.

The application allows users to submit SMS messages and receive machine learning-based risk assessments.

---

## Project Overview

SMS-based fraud is a growing concern as scammers increasingly use text messages to impersonate financial institutions, mobile money services, businesses, and trusted individuals.

Many fraudulent messages are designed to create urgency, manipulate recipients, or convince them to disclose sensitive information.

KenyaSMS Shield explores how machine learning and natural language processing can be used to detect these messages and estimate their potential risk.

Unlike a simple spam-versus-ham classifier, the system uses a two-stage classification architecture:

1. **Binary classification:** Determines whether a message is likely to be legitimate or potentially fraudulent.
2. **Risk classification:** Assigns messages to different levels of risk.

The system also incorporates Kenya-specific linguistic and structural indicators to help identify patterns commonly associated with local SMS scams.

The project includes a trained machine learning model, an interactive Streamlit application, evaluation reports, Jupyter notebooks, and an API implementation.

## Problem Statement

Traditional SMS filtering systems often rely on generic spam detection techniques. However, fraudulent messages can vary significantly in wording, structure, and context.

In Kenya, scam messages may imitate M-Pesa notifications, banking communications, customer support messages, and other familiar services.

A system trained only on generic spam patterns may fail to recognise locally relevant indicators.

This project investigates how combining textual features with manually engineered Kenya-specific indicators can improve the identification and risk assessment of suspicious SMS messages.

## Project Objectives

The main objectives are to:

1. Develop a machine learning system capable of distinguishing legitimate SMS messages from potentially fraudulent ones.
2. Implement a multi-class risk classification approach for suspicious messages.
3. Apply NLP techniques to extract meaningful information from SMS text.
4. Engineer additional features representing common Kenyan SMS scam patterns.
5. Evaluate model performance using appropriate classification metrics.
6. Incorporate explainability techniques to make model predictions more interpretable.
7. Deploy the trained model through an interactive web application.
8. Explore the practical application of machine learning in digital fraud awareness and prevention.

## Key Features

- SMS scam detection using machine learning.
- Two-stage classification architecture.
- Three-level risk assessment.
- Word-level and character-level TF-IDF feature extraction.
- Kenya-specific feature engineering.
- Logistic Regression classification.
- Model evaluation using precision, recall, F1-score, and ROC-AUC.
- Synthetic legitimate-message stress testing.
- Model explainability using SHAP.
- Interactive Streamlit application.
- FastAPI implementation.
- Reusable model bundle for inference.
- Publicly accessible web deployment.

## System Architecture

KenyaSMS Shield uses a two-stage classification architecture.

### Stage 1: Binary Classification

The first model evaluates an incoming SMS and determines whether it is likely to be legitimate or suspicious.

The output is a continuous decision score used by the system's risk-assessment logic.

### Stage 2: Risk Classification

The second model classifies the message into one of three categories:

- SAFE
- SUSPICIOUS
- HIGH-RISK SCAM

The second-stage model provides additional risk information rather than relying exclusively on the binary prediction.

### Prediction Workflow

```text
                 Incoming SMS
                      |
                      v
              Text Preprocessing
                      |
                      v
               Feature Extraction
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        Word       Character   Handcrafted
        TF-IDF     TF-IDF      Features
          |           |           |
          +-----------+-----------+
                      |
                      v
              Feature Combination
                      |
                      v
             Binary Classification
                      |
                      v
               Risk Assessment
                      |
                      v
             Severity Classification
                      |
                      v
               Final Risk Category
                      |
                      v
               User-Facing Result
```

## Risk Classification

The system produces three risk categories.

| Category | Description |
|---|---|
| SAFE | The message has a relatively low predicted scam risk. |
| SUSPICIOUS | The message contains characteristics that warrant caution. |
| HIGH-RISK SCAM | The message has a high predicted scam risk. |

The final classification uses the binary model's score, a configured decision threshold, and the severity model's prediction.

The classification is intended to support risk awareness, not to establish with certainty whether a message is fraudulent.

## Dataset and Data Preparation

The project uses a labelled SMS dataset containing legitimate and scam-related messages.

The initial dataset contained:

- 736 raw records.
- 643 cleaned unique messages.
- 376 scam messages.
- 267 legitimate messages.

Data preparation involved:

- Inspecting the dataset structure.
- Identifying missing values.
- Removing duplicate messages.
- Cleaning and standardising text.
- Preparing labels for classification.
- Creating reproducible training and testing splits.
- Separating synthetic development and holdout messages.

### Data Splitting

The original dataset was divided into training and testing subsets using a saved split.

The original held-out test set contained 128 messages.

Synthetic legitimate messages were also incorporated into model development and used for an additional false-alarm stress test.

The synthetic holdout set was kept separate from model fitting.

Synthetic data is not a substitute for independent real-world validation, and its results should be interpreted accordingly.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the characteristics of the dataset before model development.

Areas of investigation included:

- Distribution of legitimate and scam messages.
- Message length patterns.
- Word frequency distributions.
- Common terms associated with different classes.
- Text characteristics and class imbalance.
- Potentially useful patterns for feature engineering.

EDA helped inform the preprocessing and feature extraction decisions used in subsequent modelling stages.

## Feature Engineering

The system combines three feature families.

### 1. Word-Level TF-IDF

Word-level Term Frequency-Inverse Document Frequency (TF-IDF) captures the importance of individual words and word combinations.

Configuration:

- Unigrams and bigrams.
- Sublinear term frequency scaling.
- Minimum document frequency of 2.

This helps identify suspicious language patterns and common scam-related expressions.

### 2. Character-Level TF-IDF

Character-level TF-IDF captures patterns within sequences of characters.

Configuration:

- Character n-grams.
- N-gram range of 2–5.
- Sublinear term frequency scaling.
- Maximum of 50,000 features.

Character-level features can help capture variations in spelling, unusual formatting, and patterns that may not be identified through word-level analysis alone.

### 3. Handcrafted and Kenya-Specific Features

The project also extracts manually engineered features to capture structural and contextual patterns.

Examples include:

- M-Pesa-related terminology.
- M-Pesa reference patterns.
- Kenyan banking terminology.
- Money transfer language.
- Transaction reversal requests.
- Requests for credentials or sensitive information.
- Identity impersonation indicators.
- Suspicious top-level domains.
- WhatsApp and Telegram redirection patterns.
- Structured M-Pesa receipt patterns.

These features provide additional context beyond conventional text vectorisation.

### Feature Combination

The three feature families are combined using Scikit-learn's `FeatureUnion`.

| Feature family | Weight |
|---|---:|
| Word TF-IDF | 1.0 |
| Character TF-IDF | 0.8 |
| Handcrafted features | 0.7 |

The combined feature representation is passed to the classification model.

## Model Development

### Primary Algorithm: Logistic Regression

Logistic Regression was selected as the primary classification algorithm.

It is suitable for high-dimensional sparse text representations and offers relatively efficient training and inference.

The configured model uses:

- `C = 4.0`
- `max_iter = 3000`
- `class_weight = balanced`
- `random_state = 42`

The binary and multi-class models use different solver configurations appropriate to their respective classification tasks.

### Why Logistic Regression?

The choice was informed by several considerations:

1. It works effectively with sparse TF-IDF representations.
2. It provides a strong and interpretable baseline for text classification.
3. It supports efficient training and inference.
4. Its linear structure makes coefficient-based explanations possible.
5. It can handle high-dimensional feature spaces.

### Binary Classification

The first model distinguishes legitimate messages from scam messages.

A decision threshold of `0.32` was selected for the binary classification stage.

This threshold is a configured operating point, not a universally optimal threshold.

### Multi-Class Classification

The second model predicts the three risk categories.

It provides additional severity information that is used alongside the binary model's output.

## Model Evaluation

The system was evaluated using the original held-out test set.

### Binary Classification Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9401 |
| Precision | 0.8391 |
| Recall | 0.9733 |
| F1-score | 0.9012 |

### Multi-Class Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| SAFE | 0.8868 | 0.8868 | 0.8868 | 53 |
| SUSPICIOUS | 0.8667 | 0.8125 | 0.8387 | 16 |
| HIGH-RISK SCAM | 0.8833 | 0.8983 | 0.8908 | 59 |
| **Macro Average F1** | — | — | **0.8721** | **128** |

### Synthetic Legitimate-Message Stress Test

The model was additionally evaluated using a separate set of 40 synthetic legitimate messages.

| Metric | Result |
|---|---:|
| Synthetic legitimate messages | 40 |
| False alarms | 0 |
| Observed false-alarm rate | 0% |

This is a limited synthetic stress test. It should not be interpreted as evidence that the system will achieve a zero false-positive rate on real-world messages.

### Evaluation Interpretation

The binary model achieved a high recall of approximately 97.33% on the original held-out test set.

This indicates that it identified a large proportion of the scam messages in that evaluation sample.

The multi-class model achieved a macro-F1 score of approximately 0.8721.

The SUSPICIOUS category had a smaller test support of 16 messages, so its performance estimate is less stable than those of the other classes.

All reported results should be interpreted in the context of the dataset size and composition.

## Explainable AI

Model explainability is an important component of KenyaSMS Shield.

Fraud detection systems should provide more than a classification label. Understanding which features influenced a prediction can help users and developers investigate model behaviour.

The project includes a dedicated SHAP explainability notebook.

### SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to investigate feature contributions to model predictions.

The explainability analysis covers:

**Global explanations**

- Identifying influential features across multiple messages.
- Comparing feature contributions.
- Investigating patterns associated with different risk categories.

**Local explanations**

- Examining individual SMS predictions.
- Identifying features that push a prediction towards or away from a class.
- Comparing explanations for legitimate and suspicious messages.

### Explainability Visualisations

The dedicated explainability notebook includes:

- SHAP beeswarm plots.
- SHAP feature importance bar plots.
- Class-specific feature importance.
- Local waterfall plots.
- Feature contribution tables.
- Feature-family contribution analysis.

The analysis investigates the binary and severity models separately.

SHAP explanations describe model behaviour; they do not establish that a message is objectively fraudulent or that a feature represents a causal reason for fraud.

## Application Interface

KenyaSMS Shield includes an interactive web application built with Streamlit and deployed using Streamlit Community Cloud.

**Live application:** [https://kenya-sms-shield.streamlit.app/](https://kenya-sms-shield.streamlit.app/)

Users can:

1. Enter or select an SMS message.
2. Submit the message for analysis.
3. View the predicted risk category.
4. Review the system's risk assessment.
5. Access interpretation and safety guidance.

The application uses the trained machine learning model to analyse submitted messages and return predictions through an interactive interface.

The deployed application makes the system accessible without requiring users to install Python or configure a local development environment.

For developers who want to run the application locally, follow the installation instructions below.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Natural Language Processing | TF-IDF, word and character n-grams |
| Feature engineering | Custom Python transformers |
| Explainable AI | SHAP |
| Data visualisation | Matplotlib, Seaborn |
| Interactive application | Streamlit |
| API | FastAPI |
| Model persistence | Joblib |
| Development environment | Jupyter Notebook, VS Code |
| Testing | Pytest |

## Project Structure

```text
kenya-sms-shield/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
├── Dockerfile
│
├── data/
│   ├── scam_detection_labeled.csv
│   ├── split.csv
│   └── synthetic/
│
├── models/
│   └── kenya_sms_shield.joblib
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   ├── 05_optional_three_class.ipynb
│   └── 06_model_explainability_shap.ipynb
│
├── src/
│   ├── __init__.py
│   ├── advanced.py
│   ├── evaluation.py
│   ├── make_figures.py
│   ├── predict.py
│   └── preprocessing.py
│
├── reports/
│   ├── advanced_evaluation.json
│   ├── advanced_methodology.md
│   └── explainability/
│
└── tests/
```

*Note: Some generated files and optional directories may only appear after running the corresponding notebooks or scripts. The exact directory contents depend on the project version.*

## Installation and Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or a compatible version supported by the project's dependencies.
- Git.
- Visual Studio Code or another Python IDE.
- pip.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/kenya-sms-shield.git
```

### 2. Navigate to the Project Directory

```bash
cd kenya-sms-shield
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to access the application.

The trained model bundle is included in the project, so you do not need to retrain the model before running the application.

### Retraining the Model

If you want to retrain the models, run:

```bash
python -m src.advanced
```

This executes the advanced training and evaluation workflow.

## Running the API

The project also includes a FastAPI implementation.

Start the API using:

```bash
uvicorn api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation can be accessed through:

```text
http://127.0.0.1:8000/docs
```

## Running the Notebooks

Launch Jupyter Notebook from the project root:

```bash
jupyter notebook
```

The notebooks are organised to follow the project development workflow:

1. Data auditing and preparation.
2. Exploratory Data Analysis.
3. Feature engineering.
4. Model development and evaluation.
5. Optional three-class analysis.
6. SHAP model explainability.

Run the notebooks in sequence where their dependencies require it.

## Limitations

Despite its capabilities, KenyaSMS Shield has several limitations.

1. **Dataset size:** The original cleaned dataset contains 643 unique messages, which limits the diversity of language and scam patterns the model can learn.

2. **Dataset representation:** The training data may not fully represent the variety of SMS messages encountered in real-world Kenyan communication.

3. **Limited validation:** The reported performance is based on a held-out sample from the available dataset. Independent real-world validation is still necessary.

4. **Synthetic data:** Synthetic legitimate messages provide an additional stress test but cannot replace naturally collected messages.

5. **Severity classification:** The SUSPICIOUS class has relatively limited test support, making its performance estimate less stable.

6. **Language variation:** The system may struggle with slang, code-switching, unusual abbreviations, and messages in languages or writing styles underrepresented in the training data.

7. **Uncalibrated scores:** The model's output scores should not be interpreted as calibrated probabilities of fraud.

8. **External verification:** The current system does not independently verify sender identities, transaction references, website ownership, or live URL reputation.

9. **Model explainability:** Explanations indicate which features influenced model predictions, not whether those features establish fraud.

## Future Improvements

Potential improvements include:

- Expanding the dataset with real-world, responsibly collected Kenyan SMS messages.
- Incorporating additional Kenyan languages and code-switching patterns.
- Conducting independent external validation.
- Exploring alternative models such as Linear SVM, XGBoost, and transformer-based NLP models.
- Evaluating probability calibration.
- Improving risk-threshold selection through cost-sensitive evaluation.
- Incorporating URL reputation and domain intelligence.
- Exploring temporal validation to assess performance on newer scam patterns.
- Adding feedback mechanisms for users to report incorrect predictions.
- Investigating adversarial robustness against deliberately modified scam messages.
- Improving deployment through containerisation and cloud hosting.

## Disclaimer

KenyaSMS Shield is an educational and research-oriented machine learning project.

Its predictions are intended to provide additional awareness when evaluating potentially suspicious SMS messages.

The system does not guarantee the detection of all fraudulent messages, and a SAFE classification does not establish that a message is genuine.

Users should independently verify suspicious requests through official communication channels, particularly before sharing personal information, entering credentials, or making financial transactions.

## Author

**Emmilio Burure**

Data Science | Machine Learning | Natural Language Processing

- **GitHub:** [burureemmilio](https://github.com/burureemmilio)
- **LinkedIn:** [Emmilio Burure](https://www.linkedin.com/in/emmilio-burure-17b7963b4/)

---

*KenyaSMS Shield — Applying Machine Learning and Explainable AI to SMS scam detection in Kenya.*
