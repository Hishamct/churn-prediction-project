from unittest.mock import patch
from app.main import model

@patch.object(model, "predict")
def test_mock_prediction(mock_predict):

    mock_predict.return_value = [1]

    result = model.predict(
        [[1]*19]
    )

    assert result[0] == 1