import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến các thư mục gốc
cat_dir = 'E:/test/dogcatDataset/PetImages/Cat'
dog_dir = 'E:/test/dogcatDataset/PetImages/Dog'

# Đường dẫn lưu dữ liệu chia
base_dir = 'newDataSet'
os.makedirs(base_dir, exist_ok=True)

for split in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(base_dir, split, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'dog'), exist_ok=True)

def split_data(source_dir, train_dir, val_dir, test_dir, test_size=0.1, val_size=0.2):
    # Lấy tất cả các tệp tin hình ảnh trong thư mục
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42)
    
    # Sao chép ảnh vào từng thư mục
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), train_dir)
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), val_dir)
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), test_dir)

split_data(
    cat_dir,
    os.path.join(base_dir, 'train', 'cat'),
    os.path.join(base_dir, 'validation', 'cat'),
    os.path.join(base_dir, 'test', 'cat')
)

split_data(
    dog_dir,
    os.path.join(base_dir, 'train', 'dog'),
    os.path.join(base_dir, 'validation', 'dog'),
    os.path.join(base_dir, 'test', 'dog')
)

print("Dữ liệu đã được chia thành các tập train, validation và test với 1/2 bộ dữ liệu.")
