import tensorflow as tf
import time
import streamlit as st
from PIL import Image
import numpy as np
from load_css import local_css
local_css("style.css")

model = tf.keras.models.load_model('19 Des L00005_model_5_v3.h5')

st.write("""
        # Caritau nama varietas tanaman jati
        """
        )
st.write("Skripsi Achmad Makarim Widyanto")

link = '[Data Testing](https://drive.google.com/drive/folders/1EaP1X0d-E8iFykyv_5a8f_grgnAXn4Ja?usp=sharing)'
st.write("catatan: ")
st.write("1. Foto terlebih dahulu daun jati melalui aplikasi kamera")
st.write("2. File berasal dari foto yang berada di lokal atau data percobaan")
st.write("3. Silahkan unduh data uji di bawah ini untuk melakukan simulasi")
st.markdown(link, unsafe_allow_html=True)
file = st.file_uploader("Unggah file Gambar", type = ["jpg", "png"])

dimensi_gambar = (256,256)
channel = (3,)
input_shape = dimensi_gambar + channel
labels = ['mega', 'ph1', 'plus', 'tidak diketahui']

def preprocess(gambar, dimensi_gambar):
    
    #sebelum preprocess
    nimg = gambar.convert('RGB').resize(dimensi_gambar, resample= 0)
    print('sebelum preprocess:', gambar)

    #setelah preprocess
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


if file is None:
    st.text("file berbentuk jpg/png")
else:
    'Memulai pembacaan data'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
          # Update the progress bar with each iteration.
        latest_iteration.text(f'Proses... {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.1)
    gambar = Image.open(file)
    st.image(gambar, use_column_width=True)
   
    X = preprocess(gambar, dimensi_gambar)
    X = reshape([X])
    y = model.predict(X)
    
    kategori = labels[np.argmax(y)]

    'Selesai!'
    st.subheader('Hasil')
    if kategori == 'tidak diketahui':
        st.markdown(f"Varietas <span class='highlight merah'>tidak </span> diketahui",unsafe_allow_html=True)
    else :
        st.markdown(f"Tanaman jati ini termasuk varietas <span class='highlight ijo'>{kategori}</span> dengan nilai kemiripan <span class='highlight biru'> {(np.max(y)*100).astype(int)}%</span>",unsafe_allow_html=True)
        st.subheader('Nilai Probabilitas')
        st.text("(0: Mega, 1: PH1, 2: Plus, 3: Tidak Diketahui)")
        st.write(y)
#st.set_option('deprecation.showfileUploaderEncoding', False)