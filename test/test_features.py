import pytest
from main import extract_features

def test_extract_features_valid_file():
    file_path = "test_audio.wav"  # Test için bir ses dosyası yolu
    features, _, _ = extract_features(file_path)
    assert features is not None, "Öznitelikler çıkartılamadı!"
    assert len(features) == 10, "MFCC öznitelik sayısı yanlış!"

def test_extract_features_invalid_file():
    file_path = "invalid_file.wav"  # Geçersiz bir dosya yolu
    features, _, _ = extract_features(file_path)
    assert features is None, "Geçersiz dosya için None döndürülmedi!"
