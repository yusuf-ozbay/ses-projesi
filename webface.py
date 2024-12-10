import streamlit as st
import joblib
import numpy as np
import librosa
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Model ve scaler dosyasını yükle
model_tuple = joblib.load('VoiceRecognizeModel.joblib')
svc_model = model_tuple[0]
scaler = model_tuple[1]

# Gürültü azaltma fonksiyonu
def reduce_noise(audio, sr):
    return librosa.effects.remix(audio, intervals=librosa.effects.split(audio, top_db=30))

# Öznitelik çıkarma fonksiyonu
def extract_features_from_audio(audio, sample_rate):
    try:
        audio = reduce_noise(audio, sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=10)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print("Error encountered while extracting features from audio")
        return None

# Histogram çizim fonksiyonu
def plot_histogram(features, label):
    st.bar_chart(features)

# Mel spektrogramı çizim fonksiyonu
def plot_mel_spectrogram(audio, sr, label):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f"Mel-Frequency Spectrogram for {label}")
    st.pyplot(fig)

# Ses dosyasını metne dönüştürme fonksiyonu
def transcribe_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            return text
        except sr.UnknownValueError:
            st.error("Google Web Speech herhangi bir şey anlamadı")
        except sr.RequestError as e:
            st.error(f"Google Web Speech hizmetinden sonuç alınamadı; {e}")
    return ""

# Mikrofondan ses kaydetme fonksiyonu
def recognize_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Konuşmanızı bekliyorum...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            st.write("Mikrofon Metin:", text)

            # Anlık ses verisini işle
            audio = np.frombuffer(audio_data.get_wav_data(), np.int16).astype(np.float32)
            audio = librosa.util.buf_to_float(audio)
            sample_rate = 16000  # Mikrofon örnekleme oranı (standart olarak 16000 Hz)

            return audio, sample_rate, text
        except sr.UnknownValueError:
            st.write("Google Web Speech herhangi bir şey anlamadı")
        except sr.RequestError as e:
            st.write(f"Google Web Speech hizmetinden sonuç alınamadı; {e}")
    return None, None, ""

# Ses dosyasını model ile tahmin etme fonksiyonu
def predict_from_file(uploaded_file):
    audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
    features = extract_features_from_audio(audio_data, sample_rate)
    if features is not None:
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = svc_model.predict(features_scaled)
        st.write(f"Tahmin Edilen Konuşmacı: {prediction[0]}")
        # FM ve ACC değerlerini hesapla ve yazdır
        decision_function = svc_model.decision_function(features_scaled)
        fm = np.max(decision_function) - np.min(decision_function)
        acc = np.mean(svc_model.predict(features_scaled) == prediction)
        st.write(f"FM Değeri: {fm}")
        st.write(f"ACC Değeri: {acc}")
        plot_histogram(features.flatten(), prediction[0])
        plot_mel_spectrogram(audio_data, sample_rate, prediction[0])

        return audio_data, sample_rate, prediction[0], transcribe_speech(uploaded_file)
    return None, None, "", ""

# Menü kısmı
st.sidebar.header("Menu")
page = st.sidebar.radio("Sayfalar", ["Ses Tanıma", "Ses Eğitimi"])

# Ses Tanıma Sayfası
if page == "Ses Tanıma":
    st.header("Ses Tanıma")
    option = st.selectbox("Bir seçenek belirleyin:", ["Bilgisayardan Ses Seç", "Mikrofondan Ses Al"])

    audio_data = None
    sample_rate = None
    text = ""
    prediction = ""
    if option == "Bilgisayardan Ses Seç":
        uploaded_file = st.file_uploader("Bir ses dosyası seçin", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            if st.button("Tahmin Et"):
                audio_data, sample_rate, prediction, text = predict_from_file(uploaded_file)
                if text:
                    st.write("Metin:", text)
                    word_count = len(text.split())
                    st.write("Kelime Sayısı:", word_count)
                else:
                    st.write("Metin:", "Ses metne dönüştürülemedi.")
                    
    elif option == "Mikrofondan Ses Al":
        if st.button("Kaydı Al"):
            audio_data, sample_rate, text = recognize_from_microphone()
            if text:
                st.write("Metin:", text)
                word_count = len(text.split())
                st.write("Kelime Sayısı:", word_count)
                features = extract_features_from_audio(audio_data, sample_rate)
                if features is not None:
                    features = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features)
                    prediction = svc_model.predict(features_scaled)
                    st.write(f"Tahmin Edilen Konuşmacı: {prediction[0]}")
                    # FM ve ACC değerlerini hesapla ve yazdır
                    decision_function = svc_model.decision_function(features_scaled)
                    fm = np.max(decision_function) - np.min(decision_function)
                    acc = np.mean(svc_model.predict(features_scaled) == prediction)
                    st.write(f"FM Değeri: {fm}")
                    st.write(f"ACC Değeri: {acc}")
                    plot_histogram(features.flatten(), "Mikrofon Kaydı")
                    plot_mel_spectrogram(audio_data, sample_rate, "Mikrofon Kaydı")

            else:
                st.write("Metin:", "Ses metne dönüştürülemedi.")

# Ses Eğitimi Sayfası
elif page == "Ses Eğitimi":
    st.header("Ses Eğitimi")
    option = st.selectbox("Bir seçenek belirleyin:", ["Bilgisayardan Ses Seç", "Mikrofondan Ses Al"])

    name = st.text_input("Ses Sahibinin İsmi:")

    def send_to_training(audio_data, sample_rate, name):
        features = extract_features_from_audio(audio_data, sample_rate)
        if features is not None:
            st.write(f"Ses sahibinin ismi: {name}")
            plot_histogram(features, name)
            plot_mel_spectrogram(audio_data, sample_rate, name)
            st.success("Ses eğitime başarıyla yollandı")
        else:
            st.warning("Özellik çıkarılamadı. Lütfen geçerli bir ses dosyası seçin.")

    audio_data = None
    sample_rate = None
    text = ""
    if option == "Bilgisayardan Ses Seç":
        uploaded_file = st.file_uploader("Bir ses dosyası seçin", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            if st.button("Eğitime Yolla"):
                if name:
                    audio_data, sample_rate, prediction, text = predict_from_file(uploaded_file)
                    send_to_training(audio_data, sample_rate, name)
                else:
                    st.warning("Lütfen bir isim giriniz.")
        else:
            st.warning("Lütfen bir ses dosyası seçin.")

    elif option == "Mikrofondan Ses Al":
        if st.button("Kaydı Al"):
            audio_data, sample_rate, text = recognize_from_microphone()
            if text:
                st.write("Metin:", text)
                word_count = len(text.split())
                st.write("Kelime Sayısı:", word_count)
                features = extract_features_from_audio(audio_data, sample_rate)
                if features is not None:
                    st.session_state.audio_data = audio_data
                    st.session_state.sample_rate = sample_rate
                    st.session_state.text = text
                    plot_histogram(features, "Mikrofon Kaydı")
                    plot_mel_spectrogram(audio_data, sample_rate, "Mikrofon Kaydı")
            else:
                st.write("Metin:", "Ses metne dönüştürülemedi.")
        if st.button("Eğitime Yolla"):
            if name and 'audio_data' in st.session_state and st.session_state.audio_data is not None:
                send_to_training(st.session_state.audio_data, st.session_state.sample_rate, name)
            else:
                if not name:
                    st.warning("Lütfen bir isim giriniz.")
                else:
                    st.warning("Lütfen ses kaydı alın.")
