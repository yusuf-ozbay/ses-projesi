import librosa
import numpy as np
from main import reduce_noise

def test_reduce_noise():
    # Örnek ses dosyasını yükle
    audio, sr = librosa.load("C:/Users/ASUS/Desktop/Project/School/ses-projesi/speaker1/omer-sound1.wav", sr=None)
    reduced_audio = reduce_noise(audio, sr)

    # Sesin boş olmadığını kontrol et
    assert reduced_audio is not None, "Gürültü azaltma fonksiyonu None döndürdü."
    assert len(reduced_audio) > 0, "Gürültü azaltılmış ses boş."
    assert len(reduced_audio) == len(audio), "Gürültü azaltılmış sesin uzunluğu, orijinal sesle aynı olmalı."

    # Giriş ve çıkış seslerinin farklı olduğunu kontrol et
    assert not np.array_equal(reduced_audio, audio), "Gürültü azaltma sesi değiştirmedi."
