# Credit Card Fraud Detection — Machine Learning Pipeline

Pipeline completo de **Machine Learning para detecção de fraude em transações de cartão de crédito**, incluindo **EDA, pré-processamento, treinamento de modelo, avaliação e API de predição**.

O projeto demonstra boas práticas de **engenharia de ML**, com código modular (`src/`), testes automatizados (`pytest`), divisão estratificada de dados e estrutura pronta para integração em aplicações.

---

---

## Problema

Fraudes em cartões de crédito representam perdas financeiras significativas para instituições financeiras.
O objetivo deste projeto é construir um **pipeline capaz de identificar transações fraudulentas a partir de dados históricos**.

---

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn
* Pytest
* Logging
* Machine Learning Pipeline

---

---

## Dataset

O conjunto de dados utilizado neste projeto está disponível publicamente.

Faça o download em:

---

``https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud``

``https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud``

---

Após o download, coloque o arquivo em:

data/raw/creditcard.csv

---

## Estrutura do Projeto

---

project_02_fraud_detection
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   ├── api_predict.py
│   └── visualization.py
│
├── tests
│   └── test_smoke.py
│
├── pytest.ini
├── requirements.txt
└── README.md

---

## Pipeline de Machine Learning

O fluxo completo do projeto segue as etapas:

Data Loading
     ↓
Exploratory Data Analysis
     ↓
Preprocessing
     ↓
Train/Test Split (Stratified)
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Fraud Prediction API

---

## Como Executar o Projeto

Clone o repositório:

git clone ``https://github.com/HamiltonCesar425/project_02_fraud_detection.git``

cd project_02_fraud_detection

---

Crie o ambiente virtual:

python -m venv ml_env

Ative o ambiente:

ml_env\Scripts\activate

---

Instale as dependências:

pip install -r requirements.txt

---
Execute os testes:

pytest

---

Executar o pipeline de pré-processamento:

python -m src.preprocessing

---

---

## Testes Automatizados

O projeto utiliza **pytest** para validação do pipeline.

Executar testes com cobertura:

pytest --cov=src

---

## Code Quality

The project follows Python engineering best practices, including:

* **PEP8 code style**
* **Automated testing with pytest**
* **Test coverage analysis with pytest-cov**
* **Static code analysis using flake8**

Run tests and coverage:

pytest --cov=src

Run static code analysis:

flake8 .

## Objetivo do Projeto

Este projeto faz parte de um **portfólio de Data Science e Machine Learning Engineering**, demonstrando:

* construção de pipelines de ML
* organização modular de código
* boas práticas de engenharia em Python
* testes automatizados
* preparação de modelos para integração em aplicações

---

## Autor

Projeto desenvolvido como parte de um portfólio técnico de **Machine Learning e Data Science**.
