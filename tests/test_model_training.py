from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_model_training():
    # Basit veri seti
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = ["konusmaci1", "konusmaci1", "konusmaci2", "konusmaci2"]

    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modeli oluştur ve eğit
    model = SVC(kernel="linear")
    model.fit(X_scaled, y)

    # Tahmin yap ve doğrula
    prediction = model.predict([X_scaled[0]])
    assert prediction[0] == "konusmaci1", "Tahmin beklenen değeri döndürmedi."
