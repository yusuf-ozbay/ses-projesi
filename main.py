import os
import resampy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sounddevice as sd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import librosa.display
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
import pyaudio

## @package ses_tanima
#  Ses Tanıma Projesi
#  @version 1.0
#  @date 2023

# Pyaudio modülünün çalışıp çalışmadığını kontrol edin
print(pyaudio.__version__)

## Gürültü azaltma için bir fonksiyon
#  @param audio Ses verisi
#  @param sr Örnekleme oranı
#  @return Gürültüsü azaltılmış ses verisi
# Gürültü azaltma için bir fonksiyon tanımlayalım
def reduce_noise(audio, sr):
    return librosa.effects.remix(audio, intervals=librosa.effects.split(audio, top_db=30))


## Öznitelik çıkartmak için bir fonksiyon
#  @param file_path Ses dosyasının yolu
#  @return Öznitelik vektörü, ses verisi, örnekleme oranı

# Öznitelik çıkartmak için bir fonksiyon tanımlayalım
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        audio = reduce_noise(audio, sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean, audio, sample_rate
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None, None, None

def extract_features_from_audio(audio, sample_rate):
    try:
        audio = reduce_noise(audio, sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print("Error encountered while extracting features from audio")
        return None

def plot_histogram(features, speaker_label):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(features)), features, alpha=0.7)
    plt.title(f"MFCC Histogram for {speaker_label}")
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Mean Amplitude")
    plt.show()

def plot_mel_spectrogram(audio, sr, speaker_label):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-Frequency Spectrogram for {speaker_label}")
    plt.tight_layout()
    plt.show()

# Ses dosyasını metne dönüştürme
def transcribe_speech(audio, sample_rate):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            return text
        except sr.UnknownValueError:
            print("Google Web Speech herhangi bir şey anlamadı")
        except sr.RequestError as e:
            print(f"Google Web Speech hizmetinden sonuç alınamadı; {e}")
    return ""

# Ses dosyaları ve etiketleri
ses_dosyalari = [
    "konusmaci1/nurullah_train.wav", "konusmaci1/nurullah_test.wav",
    "konusmaci2/enes_train.wav", "konusmaci2/enes_test1.wav",
    "konusmaci3/voice_yasin.wav", "konusmaci3/yasin_test1.wav",
    "konusmaci4/halil_train.wav", "konusmaci4/halil_test.wav",
    "konusmaci5/hasan_train.wav", "konusmaci5/hasan_test.wav"
]

konusmaci_etiketleri = [
    "konusmaci1", "konusmaci1",
    "konusmaci2", "konusmaci2",
    "konusmaci3", "konusmaci3",
    "konusmaci4", "konusmaci4",
    "konusmaci5", "konusmaci5"
]

X = []  # Öznitelik vektörleri
y = []  # Etiketler

# Örnek bir döngü ile veri setini dolaşalım
for file_path, label in zip(ses_dosyalari, konusmaci_etiketleri):
    print("Dosya Yolu:", file_path)
    # Öznitelikleri çıkart
    features, audio, sample_rate = extract_features(file_path)
    if features is not None:
        print("Çıkarılan Öznitelikler:", features)
        # Öznitelik vektörünü X listesine ekle
        X.append(features)
        # Konuşmacının etiketini y listesine ekle
        y.append(label)
        # Konuşmacının histogramını oluştur
        plot_histogram(features, label)
        # Mel frekans spektrogramını oluştur
        plot_mel_spectrogram(audio, sample_rate, file_path)
        print("X Listesi:", X)
        print("y Listesi:", y)
    print("---------------------------------")

# Her sınıf için en az bir örnek gerekliliği
test_size = max(0.2, 3 / len(y))

konuşmacı_indeksleri = defaultdict(list)
for idx, label in enumerate(konusmaci_etiketleri):
    konuşmacı_indeksleri[label].append(idx)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = [], [], [], []
for konuşmacı, indeksler in konuşmacı_indeksleri.items():
    train_size = int(len(indeksler) * (1 - test_size))
    train_indeksler = indeksler[:train_size]
    test_indeksler = indeksler[train_size:]
    X_train.extend([X[i] for i in train_indeksler])
    X_test.extend([X[i] for i in test_indeksler])
    y_train.extend([y[i] for i in train_indeksler])
    y_test.extend([y[i] for i in test_indeksler])

# Destek Vektör Makineleri (SVM) sınıflandırıcısını seçme ve eğitme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Eğitim ve test setlerindeki doğruluk skorunu yazdır
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Test setindeki tahminleri al
y_pred = model.predict(X_test_scaled)

# Tahminleri ve gerçek etiketleri karşılaştırın
print(f"Gerçek Etiketler: {y_test}")
print(f"Tahmin Edilen Etiketler: {y_pred}")

# Sınıflandırma raporunu yazdırın
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Modelin doğruluğunu test etme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print("Model Doğruluğu:", accuracy)
print("Model Precision (Kesinlik):", precision)
print("Model Recall (Duyarlılık):", recall)
print("Model F1 Skoru:", f1)

# Mikrofondan anlık konuşmayı tanımlama
def recognize_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Konuşmanızı bekliyorum...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            print("Mikrofon Metin:", text)

            # Anlık ses verisini işle
            audio = np.frombuffer(audio_data.get_wav_data(), np.int16).astype(np.float32)
            audio = librosa.util.buf_to_float(audio)
            sample_rate = 16000  # Mikrofon örnekleme oranı (standart olarak 16000 Hz)

            features = extract_features_from_audio(audio, sample_rate)
            if features is not None:
                features = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                print(f"Tahmin Edilen Konuşmacı: {prediction[0]}")
                plot_histogram(features.flatten(), prediction[0])
                plot_mel_spectrogram(audio, sample_rate, prediction[0])

                # Kelime sayma
                word_count = len(text.split())
                print("Kelime Sayısı:", word_count)
        except sr.UnknownValueError:
            print("Google Web Speech herhangi bir şey anlamadı")
        except sr.RequestError as e:
            print(f"Google Web Speech hizmetinden sonuç alınamadı; {e}")

# Anlık konuşmayı tanımla
recognize_from_microphone()

import joblib

# Save the model and scaler
joblib.dump((model, scaler), 'VoiceRecognizeModel.joblib')

