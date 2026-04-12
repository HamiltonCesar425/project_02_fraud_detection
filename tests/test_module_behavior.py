import importlib
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api_predict, data_loading, eda, evaluate, feature_integration
from src import graph_modeling, graph_visualization, preprocessing, visualization


class DummyFigureSaver:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)



def test_load_data_empty_csv_raises_value_error():
    csv_path = Path("tests") / "_tmp_empty.csv"
    pd.DataFrame(columns=["Time", "Amount", "Class"]).to_csv(csv_path, index=False)

    try:
        with pytest.raises(ValueError, match="vazio"):
            data_loading.load_data(str(csv_path))
    finally:
        if csv_path.exists():
            csv_path.unlink()



def test_load_data_removes_duplicates():
    csv_path = Path("tests") / "_tmp_dupes.csv"
    df = pd.DataFrame(
        {
            "Time": [1, 1, 2],
            "Amount": [10.0, 10.0, 20.0],
            "Class": [0, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    try:
        result = data_loading.load_data(str(csv_path))
    finally:
        if csv_path.exists():
            csv_path.unlink()

    assert len(result) == 2



def test_load_data_warns_when_expected_columns_missing(monkeypatch):
    csv_path = Path("tests") / "_tmp_missing_cols.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(csv_path, index=False)
    warnings = []
    monkeypatch.setattr(data_loading.logger, "warning", lambda message: warnings.append(message))

    try:
        result = data_loading.load_data(str(csv_path))
    finally:
        if csv_path.exists():
            csv_path.unlink()

    assert not result.empty
    assert warnings



def test_load_data_reraises_csv_read_errors(monkeypatch):
    monkeypatch.setattr(data_loading.os.path, "exists", lambda path: True)

    def boom(path):
        raise RuntimeError("csv quebrado")

    monkeypatch.setattr(data_loading.pd, "read_csv", boom)

    with pytest.raises(RuntimeError, match="csv quebrado"):
        data_loading.load_data("fake.csv")



def test_data_loading_main_success(monkeypatch):
    called = []
    monkeypatch.setattr(data_loading.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        data_loading, "load_data", lambda path: called.append(path) or pd.DataFrame()
    )

    assert data_loading.main(["arquivo.csv"]) == 0
    assert called == ["arquivo.csv"]



def test_data_loading_main_returns_1_when_file_missing(monkeypatch):
    monkeypatch.setattr(data_loading.os.path, "exists", lambda path: False)

    assert data_loading.main(["arquivo.csv"]) == 1



def test_data_loading_main_returns_1_when_load_fails(monkeypatch):
    monkeypatch.setattr(data_loading.os.path, "exists", lambda path: True)

    def boom(path):
        raise ValueError("falha")

    monkeypatch.setattr(data_loading, "load_data", boom)

    assert data_loading.main(["arquivo.csv"]) == 1



def test_preprocess_data_missing_class_raises_value_error():
    df = pd.DataFrame({"feature1": [1, 2, 3]})

    with pytest.raises(ValueError, match="Class"):
        preprocessing.preprocess_data(df)



def test_preprocess_data_drops_nulls_before_split():
    df = pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "feature2": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "Class": [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        }
    )

    X_train, X_test, y_train, y_test = preprocessing.preprocess_data(df)

    assert len(X_train) + len(X_test) == 11
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert len(y_train) + len(y_test) == 11



def test_save_processed_dataset_writes_expected_path(monkeypatch):
    created_dirs = []
    written = []
    monkeypatch.setattr(
        preprocessing.os,
        "makedirs",
        lambda path, exist_ok: created_dirs.append((path, exist_ok)),
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, index=False: written.append((path, index)),
    )

    preprocessing.save_processed_dataset(pd.DataFrame({"a": [1]}))

    assert created_dirs == [(os.path.join("data", "processed"), True)]
    assert written == [
        (os.path.join("data", "processed", "transactions_processed.csv"), False)
    ]



def test_preprocessing_main_runs_pipeline(monkeypatch):
    source_df = pd.DataFrame({"feature": [1, 2, 3, 4], "Class": [0, 0, 1, 1]})
    fake_module = type("M", (), {})()
    fake_module.load_data = lambda path: source_df
    monkeypatch.setitem(sys.modules, "src.data_loading", fake_module)
    monkeypatch.setattr(preprocessing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        preprocessing, "preprocess_data", lambda df: ("Xtr", "Xte", "ytr", "yte")
    )
    saved = []
    monkeypatch.setattr(
        preprocessing, "save_processed_dataset", lambda df: saved.append(df.copy())
    )

    preprocessing.main()

    assert len(saved) == 1
    assert saved[0].equals(source_df.dropna())



def test_preprocessing_main_raises_when_dataset_missing(monkeypatch):
    monkeypatch.setattr(preprocessing.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        preprocessing.main()



def test_evaluate_models_missing_dataset_raises_file_not_found(monkeypatch):
    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        evaluate.evaluate_models()



def test_evaluate_models_missing_class_raises_key_error(monkeypatch):
    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        evaluate.pd, "read_csv", lambda path: pd.DataFrame({"f1": [1, 2]})
    )

    with pytest.raises(KeyError, match="Class"):
        evaluate.evaluate_models()



def test_evaluate_models_handles_model_without_predict_proba(monkeypatch):
    df = pd.DataFrame({"f1": [0, 1], "Class": [0, 1]})

    class NoProbaModel:
        def predict(self, X):
            return np.array([0, 1])

    class FakeScaler:
        def transform(self, X):
            return X

    loaded_objects = {
        "logistic_regression_model.pkl": NoProbaModel(),
        "random_forest_model.pkl": NoProbaModel(),
        "scaler.pkl": FakeScaler(),
    }

    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: True)
    monkeypatch.setattr(evaluate.pd, "read_csv", lambda path: df.copy())
    monkeypatch.setattr(
        evaluate.joblib,
        "load",
        lambda path: loaded_objects[evaluate.os.path.basename(path)],
    )
    monkeypatch.setattr(evaluate.os, "makedirs", lambda path, exist_ok: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: None)

    result = evaluate.evaluate_models()

    assert result["roc_auc"].isnull().all()



def test_evaluate_main_success(monkeypatch):
    called = []
    monkeypatch.setattr(
        evaluate, "evaluate_models", lambda: called.append(True) or pd.DataFrame()
    )

    assert evaluate.main() == 0
    assert called == [True]



def test_evaluate_main_returns_1_on_failure(monkeypatch):
    monkeypatch.setattr(
        evaluate, "evaluate_models", lambda: (_ for _ in ()).throw(RuntimeError("falha"))
    )

    assert evaluate.main() == 1



def test_api_root_returns_online_status():
    client = TestClient(api_predict.app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"



def test_api_predict_returns_500_when_model_missing():
    payload = [{**{f"V{i}": float(i) for i in range(1, 29)}, "Amount": 1.0}]
    client = TestClient(api_predict.app)

    original_model = api_predict.model
    api_predict.model = None
    try:
        response = client.post("/predict", json=payload)
    finally:
        api_predict.model = original_model

    assert response.status_code == 500
    assert "Modelo não carregado" in response.json()["detail"]



def test_api_predict_returns_500_when_model_prediction_fails():
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("falha interna")

    payload = [{**{f"V{i}": float(i) for i in range(1, 29)}, "Amount": 1.0}]
    client = TestClient(api_predict.app)

    original_model = api_predict.model
    api_predict.model = BrokenModel()
    try:
        response = client.post("/predict", json=payload)
    finally:
        api_predict.model = original_model

    assert response.status_code == 500
    assert "Erro na predição" in response.json()["detail"]



def test_api_predict_uses_raw_dataframe_when_preprocessing_fails(monkeypatch):
    class InspectingModel:
        def __init__(self):
            self.seen = None

        def predict_proba(self, X):
            self.seen = X.copy()
            return np.array([[0.4, 0.6]])

    payload = [{**{f"V{i}": float(i) for i in range(1, 29)}, "Amount": 1.0}]
    client = TestClient(api_predict.app)
    model = InspectingModel()

    original_model = api_predict.model
    monkeypatch.setattr(
        api_predict,
        "preprocess_data",
        lambda df: (_ for _ in ()).throw(ValueError("falhou")),
    )
    api_predict.model = model
    try:
        response = client.post("/predict", json=payload)
    finally:
        api_predict.model = original_model

    assert response.status_code == 200
    assert isinstance(model.seen, pd.DataFrame)
    assert list(model.seen.columns)[-1] == "Amount"



def test_api_predict_module_sets_model_none_when_load_fails(monkeypatch):
    import joblib

    original_load = joblib.load
    try:
        monkeypatch.setattr(
            joblib,
            "load",
            lambda path: (_ for _ in ()).throw(RuntimeError("sem modelo")),
        )
        sys.modules.pop("src.api_predict", None)
        reloaded = importlib.import_module("src.api_predict")
        assert reloaded.model is None
    finally:
        monkeypatch.setattr(joblib, "load", original_load)
        sys.modules.pop("src.api_predict", None)
        importlib.import_module("src.api_predict")



def test_generate_id_is_deterministic():
    assert feature_integration.generate_id(123.45) == feature_integration.generate_id(
        123.45
    )
    assert feature_integration.generate_id("abc") == feature_integration.generate_id(
        "abc"
    )



def test_feature_integration_main_requires_amount(monkeypatch):
    monkeypatch.setattr(
        feature_integration.pd,
        "read_csv",
        lambda path: pd.DataFrame({"V1": [0.1], "V2": [0.2]}),
    )

    with pytest.raises(ValueError, match="Amount"):
        feature_integration.main()



def test_feature_integration_main_success(monkeypatch):
    saved = []
    monkeypatch.setattr(
        feature_integration.pd,
        "read_csv",
        lambda path: pd.DataFrame(
            {"V1": [1.0, 2.0], "V2": [3.0, 4.0], "Amount": [10.0, 20.0]}
        ),
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, index=False: saved.append((path, self.copy())),
    )

    feature_integration.main()

    assert saved
    assert "sender_id" in saved[0][1].columns
    assert "receiver_id" in saved[0][1].columns



def test_build_weighted_graph_aggregates_duplicate_edges():
    df = pd.DataFrame(
        {
            "sender_id": [1, 1, 2],
            "receiver_id": [2, 2, 3],
            "Amount": [5.0, 7.0, 3.0],
        }
    )

    graph = graph_modeling.build_weighted_graph(df)

    assert graph[1][2]["weight"] == 12.0
    assert graph[2][3]["weight"] == 3.0



def test_compute_metrics_returns_expected_columns(monkeypatch):
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c", weight=1.0)

    monkeypatch.setattr(
        graph_modeling.nx,
        "betweenness_centrality",
        lambda G, k, normalized, seed: {node: 0.1 for node in G.nodes},
    )
    monkeypatch.setattr(
        graph_modeling.nx,
        "pagerank",
        lambda G, max_iter, tol: {node: 1 / len(G.nodes) for node in G.nodes},
    )

    result = graph_modeling.compute_metrics(graph)

    assert list(result.columns) == ["entity_id", "degree", "betweenness", "pagerank"]
    assert len(result) == 3



def test_timeout_handler_raises_timeout_exception():
    with pytest.raises(graph_modeling.TimeoutException):
        graph_modeling.timeout_handler(None, None)



def test_build_weighted_graph_skips_rows_without_endpoints():
    df = pd.DataFrame({"sender_id": [1, None], "receiver_id": [2, 3], "Amount": [5, 7]})

    graph = graph_modeling.build_weighted_graph(df)

    assert graph.number_of_edges() == 1



def test_graph_modeling_main_raises_when_input_missing(monkeypatch):
    monkeypatch.setattr(graph_modeling.os.path, "exists", lambda path: False)

    with pytest.raises(FileNotFoundError):
        graph_modeling.main()



def test_graph_modeling_main_saves_metrics(monkeypatch):
    monkeypatch.setattr(graph_modeling.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        graph_modeling.pd,
        "read_csv",
        lambda path: pd.DataFrame({"sender_id": [1], "receiver_id": [2], "Amount": [5.0]}),
    )
    monkeypatch.setattr(
        graph_modeling,
        "compute_metrics",
        lambda G: pd.DataFrame(
            {"entity_id": [1], "degree": [1], "betweenness": [0], "pagerank": [0.5]}
        ),
    )
    saved = []
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: saved.append(path))
    monkeypatch.setattr(graph_modeling.time, "time", lambda: 10.0)

    graph_modeling.main()

    assert saved == [os.path.join("data", "processed", "graph_metrics.csv")]



def test_graph_modeling_main_uses_fallback_metrics_on_timeout(monkeypatch):
    monkeypatch.setattr(graph_modeling.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        graph_modeling.pd,
        "read_csv",
        lambda path: pd.DataFrame({"sender_id": [1], "receiver_id": [2], "Amount": [5.0]}),
    )
    monkeypatch.setattr(
        graph_modeling,
        "compute_metrics",
        lambda G: (_ for _ in ()).throw(graph_modeling.TimeoutException()),
    )
    saved = []
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, path, index=False: saved.append(self.copy()))

    graph_modeling.main()

    assert len(saved) == 1
    assert list(saved[0].columns) == ["entity_id", "degree", "betweenness", "pagerank"]



def test_load_graph_raises_for_empty_edges_file(monkeypatch):
    monkeypatch.setattr(graph_visualization.os.path, "exists", lambda path: True)
    monkeypatch.setattr(graph_visualization.pd, "read_csv", lambda path: pd.DataFrame())

    with pytest.raises(ValueError, match="vazio"):
        graph_visualization.load_graph("fake.csv")



def test_sample_graph_returns_copy_for_small_graph():
    graph = nx.Graph()
    graph.add_edge(1, 2)

    sampled = graph_visualization.sample_graph(graph, max_nodes=10)

    assert sampled is not graph
    assert sampled.number_of_nodes() == graph.number_of_nodes()



def test_load_graph_success(monkeypatch):
    monkeypatch.setattr(graph_visualization.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        graph_visualization.pd,
        "read_csv",
        lambda path: pd.DataFrame({"source": [1], "target": [2], "weight": [3.5]}),
    )

    graph = graph_visualization.load_graph("fake.csv")

    assert graph.number_of_edges() == 1
    assert graph[1][2]["weight"] == 3.5



def test_sample_graph_limits_large_graph(monkeypatch):
    graph = nx.complete_graph(6)
    monkeypatch.setattr(
        graph_visualization.random,
        "sample",
        lambda nodes, max_nodes: list(nodes)[:max_nodes],
    )

    sampled = graph_visualization.sample_graph(graph, max_nodes=3)

    assert sampled.number_of_nodes() == 3



def test_plot_graph_structure_saves_figure(monkeypatch):
    saved = []
    monkeypatch.setattr(
        graph_visualization.nx,
        "spring_layout",
        lambda G, seed, k, iterations: {node: (0, 0) for node in G.nodes},
    )
    monkeypatch.setattr(graph_visualization.nx, "draw", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_visualization.plt, "savefig", lambda path, dpi=300: saved.append(path)
    )

    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph_visualization.plot_graph_structure(graph, "Titulo", "arquivo.png")

    assert saved == [os.path.join(graph_visualization.FIG_DIR, "arquivo.png")]



def test_plot_centrality_distribution_saves_figure(monkeypatch):
    saved = []
    monkeypatch.setattr(
        graph_visualization.nx, "degree_centrality", lambda G: {1: 0.5, 2: 0.5}
    )
    monkeypatch.setattr(
        graph_visualization.plt, "savefig", lambda path, dpi=300: saved.append(path)
    )

    graph = nx.Graph()
    graph.add_edge(1, 2)
    graph_visualization.plot_centrality_distribution(graph, "centrality.png")

    assert saved == [os.path.join(graph_visualization.FIG_DIR, "centrality.png")]



def test_graph_visualization_main_returns_early_when_load_fails(monkeypatch):
    called = {"sample": False}
    monkeypatch.setattr(
        graph_visualization,
        "load_graph",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("sem arquivo")),
    )
    monkeypatch.setattr(
        graph_visualization,
        "sample_graph",
        lambda G, max_nodes=500: called.__setitem__("sample", True),
    )

    graph_visualization.main()

    assert called["sample"] is False



def test_graph_visualization_main_returns_early_when_plot_fails(monkeypatch):
    monkeypatch.setattr(graph_visualization, "load_graph", lambda path: nx.Graph())
    monkeypatch.setattr(graph_visualization, "sample_graph", lambda G, max_nodes=500: G)
    monkeypatch.setattr(
        graph_visualization,
        "plot_graph_structure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("falha")),
    )

    graph_visualization.main()



def test_graph_visualization_main_success(monkeypatch):
    called = []
    monkeypatch.setattr(graph_visualization, "load_graph", lambda path: nx.Graph())
    monkeypatch.setattr(graph_visualization, "sample_graph", lambda G, max_nodes=500: G)
    monkeypatch.setattr(
        graph_visualization,
        "plot_graph_structure",
        lambda *args, **kwargs: called.append("structure"),
    )
    monkeypatch.setattr(
        graph_visualization,
        "plot_centrality_distribution",
        lambda *args, **kwargs: called.append("centrality"),
    )

    graph_visualization.main()

    assert called == ["structure", "centrality"]



def test_plot_model_performance_skips_when_metrics_file_missing(monkeypatch):
    monkeypatch.setattr(visualization.os.path, "exists", lambda path: False)
    called = {"savefig": False}
    monkeypatch.setattr(
        visualization.plt,
        "savefig",
        lambda *args, **kwargs: called.__setitem__("savefig", True),
    )

    visualization.plot_model_performance("missing.csv")

    assert called["savefig"] is False



def test_plot_distribution_saves_figure(monkeypatch):
    saver = DummyFigureSaver()
    monkeypatch.setattr(visualization.plt, "savefig", saver)

    df = pd.DataFrame({"Amount": [10, 20, 30]})
    visualization.plot_distribution(df, "Amount", "Distribuicao")

    assert saver.paths
    assert saver.paths[0].endswith("Amount_distribution.png")



def test_plot_model_performance_saves_figure(monkeypatch):
    saver = DummyFigureSaver()
    monkeypatch.setattr(visualization.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        visualization.pd,
        "read_csv",
        lambda path: pd.DataFrame({"model": ["m1"], "roc_auc": [0.9]}),
    )
    monkeypatch.setattr(visualization.plt, "savefig", saver)

    visualization.plot_model_performance("metrics.csv")

    assert saver.paths[0].endswith("model_performance.png")



def test_visualization_main_runs_both_sections(monkeypatch):
    calls = []
    monkeypatch.setattr(
        visualization.pd,
        "read_csv",
        lambda path: pd.DataFrame({"Amount": [1], "Time": [2]}),
    )
    monkeypatch.setattr(
        visualization, "plot_distribution", lambda df, column, title: calls.append(column)
    )
    monkeypatch.setattr(
        visualization, "plot_model_performance", lambda path: calls.append("performance")
    )

    visualization.main()

    assert calls == ["Amount", "Time", "performance"]



def test_visualization_main_handles_distribution_failure(monkeypatch):
    monkeypatch.setattr(
        visualization.pd,
        "read_csv",
        lambda path: (_ for _ in ()).throw(RuntimeError("falha")),
    )
    called = []
    monkeypatch.setattr(visualization, "plot_model_performance", lambda path: called.append(path))

    visualization.main()

    assert called == [os.path.join("reports", "model_metrics.csv")]



def test_visualization_main_handles_performance_failure(monkeypatch):
    monkeypatch.setattr(
        visualization.pd,
        "read_csv",
        lambda path: pd.DataFrame({"Amount": [1], "Time": [2]}),
    )
    monkeypatch.setattr(visualization, "plot_distribution", lambda df, column, title: None)
    monkeypatch.setattr(
        visualization,
        "plot_model_performance",
        lambda path: (_ for _ in ()).throw(RuntimeError("falha")),
    )

    visualization.main()



def test_eda_missing_values_report_returns_counts():
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})

    report = eda.missing_values_report(df)

    assert report.loc["a", "missing_count"] == 1
    assert report.loc["b", "missing_count"] == 2



def test_eda_detect_outliers_flags_extreme_values():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 100]})

    outliers = eda.detect_outliers(df)

    assert outliers.loc["x", "outlier_count"] == 1



def test_eda_load_data_uses_external_loader(monkeypatch):
    fake_module = type("M", (), {})()
    fake_module.load_data = lambda path: pd.DataFrame({"x": [1]})
    monkeypatch.setitem(sys.modules, "data_loading", fake_module)

    result = eda.load_data("arquivo.csv")

    assert list(result.columns) == ["x"]



def test_eda_load_data_returns_none_when_file_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "data_loading", raising=False)
    monkeypatch.setattr(eda.os.path, "exists", lambda path: False)

    assert eda.load_data("arquivo.csv") is None



def test_eda_load_data_falls_back_to_read_csv(monkeypatch):
    fake_module = type("M", (), {"load_data": lambda path: None})()
    monkeypatch.setitem(sys.modules, "data_loading", fake_module)
    monkeypatch.setattr(eda.os.path, "exists", lambda path: True)
    monkeypatch.setattr(eda.pd, "read_csv", lambda path: pd.DataFrame({"fallback": [1]}))

    result = eda.load_data("arquivo.csv")

    assert list(result.columns) == ["fallback"]



def test_eda_basic_info_logs_shape_and_dtypes(monkeypatch):
    messages = []
    monkeypatch.setattr(eda.logger, "info", lambda message: messages.append(message))

    eda.basic_info(pd.DataFrame({"a": [1]}))

    assert any("Shape:" in msg for msg in messages)



def test_eda_target_distribution_without_target_returns_early(monkeypatch):
    messages = []
    monkeypatch.setattr(eda.logger, "error", lambda message: messages.append(message))

    eda.target_distribution(pd.DataFrame({"x": [1, 2]}), target_col="Class", out_dir="tests")

    assert messages



def test_eda_target_distribution_saves_figure(monkeypatch):
    saved = []
    monkeypatch.setattr(eda, "save_figure", lambda path: saved.append(path))

    eda.target_distribution(pd.DataFrame({"Class": [0, 1, 1]}), out_dir="tests")

    assert saved == [os.path.join("tests", "target_distribution.png")]



def test_eda_target_distribution_shows_when_no_out_dir(monkeypatch):
    shown = []
    monkeypatch.setattr(eda.plt, "show", lambda: shown.append(True))

    eda.target_distribution(pd.DataFrame({"Class": [0, 1, 1]}), out_dir=None)

    assert shown == [True]



def test_eda_plot_correlation_matrix_saves_figure(monkeypatch):
    saved = []
    monkeypatch.setattr(eda, "save_figure", lambda path: saved.append(path))

    eda.plot_correlation_matrix(pd.DataFrame({"a": [1, 2], "b": [2, 3]}), out_dir="tests")

    assert saved == [os.path.join("tests", "correlation_matrix.png")]



def test_eda_plot_correlation_matrix_shows_when_no_out_dir(monkeypatch):
    shown = []
    monkeypatch.setattr(eda.plt, "show", lambda: shown.append(True))

    eda.plot_correlation_matrix(pd.DataFrame({"a": [1, 2], "b": [2, 3]}), out_dir=None)

    assert shown == [True]



def test_eda_plot_feature_distributions_saves_each_feature(monkeypatch):
    saved = []
    monkeypatch.setattr(eda, "save_figure", lambda path: saved.append(path))

    eda.plot_feature_distributions(
        pd.DataFrame({"a": [1, 2], "b": [3, 4], "Class": [0, 1]}),
        out_dir="tests",
        max_plots=2,
    )

    assert saved == [
        os.path.join("tests", "feature_1_a.png"),
        os.path.join("tests", "feature_2_b.png"),
    ]



def test_eda_plot_feature_distributions_shows_without_out_dir(monkeypatch):
    shown = []
    monkeypatch.setattr(eda.plt, "show", lambda: shown.append(True))

    eda.plot_feature_distributions(
        pd.DataFrame({"a": [1, 2], "Class": [0, 1]}), out_dir=None, max_plots=1
    )

    assert shown == [True]



def test_eda_save_figure_creates_directory_and_saves(monkeypatch):
    ensured = []
    saved = []
    monkeypatch.setattr(eda, "_ensure_dir", lambda path: ensured.append(path))
    monkeypatch.setattr(
        eda.plt,
        "savefig",
        lambda path, bbox_inches="tight": saved.append((path, bbox_inches)),
    )

    eda.save_figure(os.path.join("tests", "figs", "plot.png"))

    assert ensured == [os.path.join("tests", "figs")]
    assert saved == [(os.path.join("tests", "figs", "plot.png"), "tight")]



def test_eda_run_eda_aborts_when_load_data_fails(monkeypatch):
    called = {"basic_info": False}
    monkeypatch.setattr(eda, "load_data", lambda path: None)
    monkeypatch.setattr(eda, "basic_info", lambda df: called.__setitem__("basic_info", True))

    eda.run_eda("missing.csv", out_dir="tests")

    assert called["basic_info"] is False



def test_eda_run_eda_success(monkeypatch):
    df = pd.DataFrame({"Time": [1, 2], "Amount": [10, 20], "Class": [0, 1]})
    called = []
    saved_csv = []
    monkeypatch.setattr(eda, "load_data", lambda path: df)
    monkeypatch.setattr(eda, "basic_info", lambda frame: called.append("basic"))
    monkeypatch.setattr(
        eda, "missing_values_report", lambda frame: called.append("missing")
    )
    monkeypatch.setattr(
        eda,
        "target_distribution",
        lambda frame, target_col, out_dir: called.append(
            ("target", list(frame.columns), target_col, out_dir)
        ),
    )
    monkeypatch.setattr(
        eda,
        "plot_correlation_matrix",
        lambda frame, out_dir: called.append(("corr", list(frame.columns), out_dir)),
    )
    monkeypatch.setattr(
        eda, "detect_outliers", lambda frame: pd.DataFrame({"outlier_count": [0]}, index=["Amount"])
    )
    monkeypatch.setattr(
        eda,
        "plot_feature_distributions",
        lambda frame, out_dir=None: called.append(("features", list(frame.columns), out_dir)),
    )
    monkeypatch.setattr(
        pd.DataFrame, "to_csv", lambda self, path: saved_csv.append(path)
    )

    eda.run_eda("input.csv", out_dir="tests", target_col="Class")

    assert "basic" in called
    assert "missing" in called
    assert ("target", ["Amount", "Class"], "Class", "tests") in called
    assert ("corr", ["Amount"], "tests") in called
    assert ("features", ["Amount", "Class"], "tests") in called
    assert saved_csv == [os.path.join("tests", "outlier_summary.csv")]



def test_eda_main_parses_arguments(monkeypatch):
    captured = []
    monkeypatch.setattr(
        eda,
        "run_eda",
        lambda input_path, out_dir, target_col: captured.append(
            (input_path, out_dir, target_col)
        ),
    )

    eda.main(["--input", "dados.csv", "--out_dir", "saidas", "--target", "Fraud"])

    assert captured == [("dados.csv", "saidas", "Fraud")]
