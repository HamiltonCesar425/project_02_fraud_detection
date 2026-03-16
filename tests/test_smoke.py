from src.preprocessing import preprocess_data
import pandas as pd

def test_preprocess_returns_split_sets():
    df = pd.DataFrame({
        "amount": [100,200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "time": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        "Class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = preprocess_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0 
    assert len(y_train) > 0
    assert len(y_test) > 0
      
