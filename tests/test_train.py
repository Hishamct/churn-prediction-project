from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def test_model_training():

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(X, y)

    accuracy = model.score(X, y)

    assert accuracy > 0.5