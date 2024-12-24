import librosa
from main import plot_mel_spectrogram
#Mel Spektrogram Çizim Fonksiyonunu Test Etme
def test_plot_mel_spectrogram():
    audio, sr = librosa.load("C:/Users/ASUS/Desktop/Project/School/ses-projesi/speaker1/omer-sound1.wav", sr=None)
    try:
        plot_mel_spectrogram(audio, sr, "konusmaci1")
        assert True, "Mel spektrogram çizimi başarılı."
    except Exception as e:
        assert False, f"Mel spektrogram çizimi başarısız oldu: {e}"
