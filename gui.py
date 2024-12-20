import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('model1_catdog_10epoch.h5')

# Kết quả phân loại
results = {
    0: 'cat',
    1: 'dog'
}

Image_Size = (128, 128)  # Đảm bảo bạn có giá trị cho Image_Size

# Giao diện Streamlit
st.title('Phân loại Chó và Mèo')

uploaded_file = st.file_uploader("Tải lên ảnh của bạn", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn đã tải lên', use_column_width=True)

    if st.button('Dự đoán'):
        # Resize và chuẩn hóa ảnh
        im = image.resize(Image_Size)  # Resize ảnh theo kích thước mong muốn
        im = np.expand_dims(im, axis=0)  # Thêm chiều batch
        im = np.array(im)  # Chuyển ảnh thành mảng numpy
        im = im / 255  # Chuẩn hóa ảnh

        # Dự đoán lớp của ảnh
        pred = np.argmax(model.predict(im), axis=-1)[0]  # Dự đoán lớp với xác suất cao nhất

        # Hiển thị kết quả
        st.write(f"### Dự đoán: {results[pred]}")