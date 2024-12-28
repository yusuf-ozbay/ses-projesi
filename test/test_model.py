import pytest
from sklearn.svm import SVC
import numpy as np

@pytest.fixture
def mock_data():
    X_train = np.random.rand(10, 10)  # Rastgele 10 örnek
    y_train = ['konusmaci1', 'konusmaci2'] * 5  # 10 etiket
    return X_train, y_train

def test_model_training(mock_data):
    X_train, y_train = mock_data
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    assert model, "Model eğitimi başarısız!"
    assert model.support_vectors_.shape[0] > 0, "Destek vektörleri oluşturulamadı!"