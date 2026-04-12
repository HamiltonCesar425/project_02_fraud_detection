from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src import api_predict, data_loading, evaluate, preprocessing


def test_load_data_success():
    csv_path = Path("tests") / "_tmp_fake.csv"
    df = pd.DataFrame(
        {
            "Time": [1, 2],
            "Amount": [10.0, 20.0],
            "Class": [0, 1],
        }
    )

    df.to_csv(csv_path, index=False)
    try:
        result = data_loading.load_data(str(csv_path))
    finally:
        if csv_path.exists():
            csv_path.unlink()

    assert not result.empty
    assert {"Time", "Amount", "Class"}.issubset(result.columns)


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        data_loading.load_data("arquivo_inexistente.csv")


def test_preprocess_data_success():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "Class": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )

    X_train, X_test, y_train, y_test = preprocessing.preprocess_data(df)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert list(X_train.columns) == ["feature1", "feature2"]



def test_evaluate_models_metrics(monkeypatch):
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "Class": [0, 1, 0, 1],
        }
    )

    class FakeScaler:
        def transform(self, X):
            return X.to_numpy()

    class FakeModel:
        def predict(self, X):
            return np.array([0, 1, 0, 1])

        def predict_proba(self, X):
            return np.array(
                [
                    [0.9, 0.1],
                    [0.1, 0.9],
                    [0.8, 0.2],
                    [0.2, 0.8],
                ]
            )

    loaded_objects = {
        "logistic_regression_model.pkl": FakeModel(),
        "random_forest_model.pkl": FakeModel(),
        "scaler.pkl": FakeScaler(),
    }

    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: True)
    monkeypatch.setattr(evaluate.pd, "read_csv", lambda path: df.copy())
    monkeypatch.setattr(
        evaluate.joblib,
        "load",
        lambda path: loaded_objects[evaluate.os.path.basename(path)],
    )
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: None)

    result = evaluate.evaluate_models()

    assert list(result["model"]) == ["Logistic Regression", "Random Forest"]
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(
        result.columns
    )
    assert (result["accuracy"] == 1.0).all()



def test_api_predict_returns_json():
    payload = [
        {
            **{f"V{i}": float(i) / 10 for i in range(1, 29)},
            "Amount": 50.0,
        },
        {
            **{f"V{i}": float(i) / 5 for i in range(1, 29)},
            "Amount": 120.0,
        },
    ]

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

    client = TestClient(api_predict.app)

    original_model = api_predict.model
    api_predict.model = mock_model
    try:
        result = client.post("/predict", json=payload)
    finally:
        api_predict.model = original_model

    assert result.status_code == 200
    body = result.json()
    assert "results" in body
    assert len(body["results"]) == 2
    assert body["results"][0]["transaction_id"] == 1
    assert body["results"][0]["fraud_probability"] == 0.8
    assert body["results"][0]["fraud_prediction"] == 1
