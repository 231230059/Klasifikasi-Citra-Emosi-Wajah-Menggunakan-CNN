import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# KONFIGURASI HALAMAN & STYLE
# ------------------------------------------------------------
st.set_page_config(
    page_title='Dashboard Facial Emotion Recognition',
    page_icon='🎭',
    layout='wide',
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'output')
MODEL_PATH = os.path.join(OUT_DIR, 'final_emotion_cnn_model.keras')
CLASS_PATH = os.path.join(OUT_DIR, 'class_names.json')
IMG_SIZE = (48, 48)

# Emoji representasi emosi
EMOJI = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
}

# ------------------------------------------------------------
# LOAD MODEL & ASSET
# ------------------------------------------------------------
@st.cache_resource(show_spinner='Memuat model dan data...')
def load_assets():
  # Cek ketersediaan model
  if not os.path.exists(MODEL_PATH):
    return None, None
  model = tf.keras.models.load_model(MODEL_PATH)

  # Load nama kelas jika ada, atau gunakan default FER2013
  if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH) as f:
      class_names = json.load(f)
  else:
    class_names = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'sad',
        'surprise',
    ]

  return model, class_names


model, class_names = load_assets()

# ------------------------------------------------------------
# SIDEBAR / NAVIGASI
# ------------------------------------------------------------
st.sidebar.title('🧭 Navigasi Menu')
menu = st.sidebar.selectbox(
    'Pilih Halaman',
    [
        '🏠 Beranda & Ringkasan Proyek',
        '📊 Analisis & Evaluasi Model',
        '🔍 Uji Prediksi Gambar',
    ],
)

# ------------------------------------------------------------
# HALAMAN 1: BERANDA & RINGKASAN PROYEK
# ------------------------------------------------------------
if menu == '🏠 Beranda & Ringkasan Proyek':
  st.title('🎭 Dashboard Eksperimen CNN - Facial Emotion Recognition')
  st.markdown(
      'Aplikasi dashboard interaktif ini dikembangkan untuk memenuhi tugas'
      ' praktikum Deep Learning dalam mengenali dan mengklasifikasikan ekspresi'
      ' emosi wajah manusia menggunakan arsitektur **Convolutional Neural'
      ' Network (CNN)**.'
  )

  st.markdown('---')
  st.subheader('📌 Ringkasan Dataset & Konfigurasi')

  col1, col2, col3, col4 = st.columns(4)
  col1.metric('Total Kelas Emosi', len(class_names) if class_names else 7)
  col2.metric('Ukuran Input Citra', '48 x 48 Piksel')
  col3.metric('Proporsi Data Latih', '80%')
  col4.metric('Proporsi Data Uji', '20%')

  st.markdown('### 📋 Daftar Kelas Emosi yang Dikenali:')
  if class_names:
    cols = st.columns(len(class_names))
    for i, c in enumerate(class_names):
      with cols[i]:
        st.info(f'{EMOJI.get(c, "✨")} **{c.capitalize()}**')

# ------------------------------------------------------------
# HALAMAN 2: ANALISIS & EVALUASI MODEL
# ------------------------------------------------------------
elif menu == '📊 Analisis & Evaluasi Model':
  st.title('📊 Hasil Evaluasi & Performa Model CNN')
  st.write(
      'Halaman ini menampilkan grafik hasil pelatihan, confusion matrix, serta'
      ' ringkasan metrik evaluasi model.'
  )

  # Cek file visualisasi dari folder output
  train_curve_path = os.path.join(OUT_DIR, '2_training_curves.png')
  cm_path = os.path.join(OUT_DIR, '3_confusion_matrix.png')

  tab1, tab2 = st.tabs(
      ['📈 Grafik Pelatihan (Akurasi & Loss)', '📉 Confusion Matrix & Metrik']
  )

  with tab1:
    st.subheader('Grafik Konvergensi Pelatihan Model')
    if os.path.exists(train_curve_path):
      st.image(
          train_curve_path,
          caption='Grafik Akurasi dan Loss (Train vs Validation)',
          use_container_width=True,
      )
    else:
      st.warning(
          '⚠️ File grafik `2_training_curves.png` belum ditemukan di folder'
          ' `output/`. Jalankan skrip training di notebook terlebih dahulu.'
      )

  with tab2:
    st.subheader('Confusion Matrix Klasifikasi Emosi')
    if os.path.exists(cm_path):
      st.image(
          cm_path,
          caption='Confusion Matrix Proporsi Prediksi Model',
          use_container_width=True,
      )
    else:
      st.warning(
          '⚠️ File confusion matrix `3_confusion_matrix.png` belum ditemukan di'
          ' folder `output/`.'
      )

    st.markdown('---')
    st.subheader('📊 Ringkasan Metrik Evaluasi')
    st.markdown("""
        * **Accuracy (Akurasi):** Mengukur ketepatan keseluruhan model dalam memprediksi emosi.
        * **Precision (Presisi):** Tingkat ketepatan antara data yang diprediksi positif dibandingkan dengan data aktual.
        * **Recall (Sensitivitas):** Kemampuan model menemukan kembali suatu informasi kelas emosi tertentu.
        * **F1-Score:** Nilai keseimbangan harmonis antara Precision dan Recall.
        """)

# ------------------------------------------------------------
# HALAMAN 3: UJI PREDIKSI GAMBAR (REAL-TIME)
# ------------------------------------------------------------
elif menu == '🔍 Uji Prediksi Gambar':
  st.title('🔍 Uji Prediksi Emosi Wajah Secara Real-Time')
  st.write(
      'Unggah gambar wajah (format JPG/PNG) untuk menguji performa model CNN'
      ' secara langsung pada dashboard.'
  )

  if model is None:
    st.error(
        '❌ Model belum dimuat! Pastikan file model `.keras` atau `.h5` sudah'
        ' tersedia di dalam folder `output/`.'
    )
  else:
    uploaded_file = st.file_uploader(
        'Pilih file gambar wajah...', type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
      image = Image.open(uploaded_file)

      col1, col2 = st.columns(2, gap='large')

      with col1:
        st.image(
            image, caption='Gambar Wajah yang Diunggah', use_container_width=True
        )

      with col2:
        # Pra-pemrosesan gambar
        img_gray = image.convert('L').resize(IMG_SIZE)
        img_array = np.array(img_gray, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # (48, 48, 1)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 48, 48, 1)

        # Prediksi Model
        with st.spinner('Model sedang menganalisis ekspresi wajah...'):
          predictions = model.predict(img_array, verbose=0)[0]

        top_idx = np.argmax(predictions)
        predicted_emotion = class_names[top_idx]
        confidence = float(predictions[top_idx]) * 100

        # Tampilkan Hasil Utama
        st.success('### Hasil Prediksi Real-Time')
        st.markdown(
            f"""
                <div style="background-color: #7F8C6A; padding: 15px; border-radius: 10px; text-align: center;">
                    <h1>{EMOJI.get(predicted_emotion, '✨')} {predicted_emotion.upper()}</h1>
                    <h3>Tingkat Keyakinan: <b>{confidence:.2f}%</b></h3>
                </div>
                """,
            unsafe_allow_html=True,
        )

      st.markdown('---')
      st.subheader('📊 Distribusi Probabilitas Seluruh Kelas Emosi')
      # Tampilkan bar chart / progress bar untuk setiap kelas
      sorted_indices = np.argsort(predictions)[::-1]
      for idx in sorted_indices:
        emotion_name = class_names[idx]
        score = float(predictions[idx])
        col_name, col_prog, col_val = st.columns([2, 5, 1])
        with col_name:
          st.write(f'{EMOJI.get(emotion_name, "")} {emotion_name.capitalize()}')
        with col_prog:
          st.progress(min(max(score, 0.0), 1.0))
        with col_val:
          st.write(f'{score * 100:.1f}%')

    else:
      st.info(
          '👆 Silakan unggah gambar pada tombol di atas untuk memulai uji'
          ' prediksi.'
      )

# ------------------------------------------------------------
# FOOTER (Diperbaiki menggunakan st.markdown)
# ------------------------------------------------------------
st.markdown('---')
st.markdown(
    '<p style="text-align: center; color: gray;">Dashboard Praktikum Deep'
    ' Learning — Facial Emotion Recognition (FER2013)</p>',
    unsafe_allow_html=True,
)