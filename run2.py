from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model('model1_catdog_10epoch.h5')

img = image.load_img('./image.png', target_size=(128, 128))

results = {
    0: 'cat',
    1: 'dog'
}
Image_Size = (128, 128)  # Đảm bảo bạn có giá trị cho Image_Size
im = Image.open("./image.png")
im = im.resize(Image_Size)  # Resize ảnh theo kích thước mong muốn
im = np.expand_dims(im, axis=0)  # Thêm chiều batch
im = np.array(im)  # Chuyển ảnh thành mảng numpy
im = im / 255  # Chuẩn hóa ảnh

# Dự đoán lớp của ảnh
pred = np.argmax(model.predict(im), axis=-1)[0]  # Dự đoán lớp với xác suất cao nhất

# In kết quả
print(pred, results[pred])
