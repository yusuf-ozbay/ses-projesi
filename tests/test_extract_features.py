from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os


def test_extract_features():
    # Headless mode ayarları
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("http://localhost:8502")  # Web uygulamanızın URL'si

    # Dosya yükleme alanını bulun
    upload_field = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )

    file_path = os.path.abspath("speaker1/omer-sound1.wav")
    upload_field.send_keys(file_path)

    # Öznitelik çıkarma butonuna bas
    driver.find_element(By.ID, "extract-features-button").click()

    # Çıktıyı kontrol et
    features_output = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "features-output"))
    )
    assert "MFCC" in features_output.text, "Öznitelik çıkarma başarısız."

    driver.quit()
