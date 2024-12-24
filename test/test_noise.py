import pytest
import numpy as np
from main import reduce_noise

def test_reduce_noise_length():
    sample_audio = np.random.rand(16000)  # 1 saniyelik rastgele ses verisi
    sr = 16000  # Örnekleme oranı
    reduced_audio = reduce_noise(sample_audio, sr)
    assert len(reduced_audio) <= len(sample_audio), "Gürültü azaltma sonucu uzunluk hatası!"

def test_reduce_noise_functionality():
    sample_audio = np.zeros(16000)  # Gürültüsüz (sessiz) ses verisi
    sr = 16000
    reduced_audio = reduce_noise(sample_audio, sr)
    assert np.array_equal(reduced_audio, sample_audio), "Sessiz ses yanlış değiştirildi!"
