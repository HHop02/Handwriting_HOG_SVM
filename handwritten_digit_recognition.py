import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tải dữ liệu MNIST
digits = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
images, labels = digits['data'], digits['target'].astype(np.int64)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Tính toán HOG features
def compute_hog_features(images):
    hog_features = []
    for image in images:
        feature = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(feature)
    return np.array(hog_features)

X_train_hog = compute_hog_features(X_train)
X_test_hog = compute_hog_features(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_hog, y_train)

# Đánh giá mô hình
y_pred = svm_model.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác: {accuracy * 100:.2f}%')

# Hiển thị một số kết quả dự đoán
def display_predictions(images, true_labels, predicted_labels, num_display=20):
    for i in range(num_display):
        image = images[i].reshape((28, 28))
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true_labels[i]}, Dự đoán: {predicted_labels[i]}')
        plt.axis('off')
        plt.show()

# Hiển thị kết quả dự đoán của một số hình ảnh từ tập kiểm tra
# display_predictions(X_test, y_test, y_pred)

# nhận diện chữ viết tay từ ảnh 
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"File not found or unable to read: {image_path}")
            return None
        if len(image.shape) == 3 and image.shape[2] == 4:  # Nếu hình ảnh có 4 kênh màu
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Nếu là ảnh màu
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Đường dẫn đến hình ảnh 
image_paths = [
    
    'Test_img/img_2.png',
    'Test_img/img2_1.png',

    'Test_img/img_3.png',
    'Test_img/img3_1.png',
    'Test_img/img3_2.png',
    
    'Test_img/img_1.png',
    'Test_img/img1_1.png',
    
    'Test_img/img_4.png',
    'Test_img/img4_3.png',
    
    'Test_img/img9_1.png',
    'Test_img/img9_2.png',#!
    'Test_img/img9_5.png' ,

    'handwriting/img1.jpg',
    'handwriting/img2.jpg',
    'handwriting/img3.jpg',
    'handwriting/img4_3.jpg',
    'handwriting/img4_5.jpg',
    'handwriting/img5_2.jpg',
    'handwriting/img6_2.jpg',
    'handwriting/img7.jpg',
    'handwriting/img8.jpg',
    'handwriting/img9.jpg' 
    
    'Test_img/img_5.png',
    'Test_img/img5_1.png',#!
    'Test_img/img5_2.png',

    'Test_img/img7_1.png',
    'Test_img/img7_2.png',
    
    'Test_img/img8_4.png',
    'Test_img/img8_8.png',

    'Test_img/img6_4.png',
    'Test_img/img6_5.png',#!
  



]

# Tiền xử lý và tính toán HOG features cho ảnh 
images = [preprocess_image(img_path) for img_path in image_paths]
images = [img for img in images if img is not None]


if len(images) > 0:
    hog_features = compute_hog_features(images)
    hog_features = scaler.transform(hog_features)
    predictions = svm_model.predict(hog_features)

    # Dự đoán nhãn cho ảnh 
    predictions = svm_model.predict(hog_features)

    # Hiển thị kết quả dự đoán
    display_predictions(np.array(images), predictions, predictions)
else:
    print("No valid images found for prediction.")