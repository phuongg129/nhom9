import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import torch.nn as nn

# Kiểm tra xem file mô hình có tồn tại không
if not os.path.exists('animal_classifier.pth'):
    print("Model file not found! Training the model now...")

    # Tạo mô hình (ResNet18)
    model = models.resnet18(pretrained=True)  # Sử dụng mô hình đã được huấn luyện trước
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 lớp cho 5 loại động vật

    # Định nghĩa các phép biến đổi cho hình ảnh (bao gồm augmentations)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),  # Phép biến đổi ngẫu nhiên
        transforms.RandomRotation(10),  # Xoay hình ảnh ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Giả sử bạn có bộ dữ liệu huấn luyện
    train_dataset = datasets.ImageFolder(r'D:/animal-identification-main/animal-identification-main/data/animals/train', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Định nghĩa loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    for epoch in range(70):  # Huấn luyện trong 60 epoch
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/70: Loss = {loss.item()}')

    # Sau khi huấn luyện xong, lưu mô hình
    torch.save(model.state_dict(), 'animal_classifier.pth')
    print("Model trained and saved as 'animal_classifier.pth'.")

else:
    # Nếu file mô hình đã tồn tại, tải trạng thái của mô hình
    model = models.resnet18(pretrained=False)  # Không sử dụng weights mặc định
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 lớp cho 5 loại động vật
    # Tải trọng số của mô hình
    model.load_state_dict(torch.load('animal_classifier.pth'))
    print("Model loaded from 'animal_classifier.pth'.")

# Chuyển mô hình sang chế độ đánh giá
model.eval()

# Định nghĩa các phép biến đổi cho hình ảnh (sử dụng lại transform từ huấn luyện)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Xác định danh sách các loài động vật
class_names = ['capybara', 'chim-canh-cut', 'cho', 'ga', 'meo']

def predict_image():
    # Mở cửa sổ chọn tệp để người dùng chọn hình ảnh
    file_path = filedialog.askopenfilename()

    print(f"Selected file path: {file_path}")  # In ra đường dẫn file được chọn

    # Kiểm tra xem file có phải là hình ảnh hay không
    if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):

        try:
            # Tải hình ảnh từ đường dẫn
            img = Image.open(file_path)

            # Áp dụng các phép biến đổi cho hình ảnh
            img_t = transform(img)
            img_t = img_t.unsqueeze(0)

            # Dự đoán nhãn cho hình ảnh
            with torch.no_grad():
                output = model(img_t)
                predicted_class = output.argmax(dim=1).item()

            # Lấy nhãn được dự đoán
            predicted_label = class_names[predicted_class]

            # Thu nhỏ ảnh trước khi hiển thị
            img = img.resize((200, 200))  # Thu nhỏ ảnh về kích thước 200x200

            # Hiển thị hình ảnh trong cửa sổ chính
            img_tk = ImageTk.PhotoImage(img)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # Lưu trữ ảnh để tránh mất dữ liệu

            # Cập nhật kết quả dự đoán bằng tiếng Việt có dấu
            result_label.config(text=f'Nhãn dự đoán: {predicted_label}')

        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print("Selected file is not a valid image.")

# Tạo cửa sổ giao diện người dùng chính
root = tk.Tk()
root.title('Animal Classifier')

# Cài đặt kích thước cửa sổ
root.geometry('700x500')  # Tăng chiều rộng cửa sổ để có không gian cho cả ảnh và kết quả

# Thêm tiêu đề đẹp mắt
title_label = tk.Label(root, text="Animal Classifier", font=('Helvetica', 24, 'bold'), bg='lightblue', fg='darkblue')
title_label.pack(pady=20)

# Tạo nút chọn hình ảnh
select_button = tk.Button(root, text='Chọn Hình Ảnh', font=('Helvetica', 14), command=predict_image, bg='orange', fg='white')
select_button.pack(pady=10)

# Tạo một frame để chứa ảnh và kết quả
frame = tk.Frame(root)
frame.pack(pady=20)

# Tạo label để hiển thị hình ảnh
img_label = tk.Label(frame)
img_label.grid(row=0, column=0, padx=20)  # Đặt ảnh bên trái

# Tạo label để hiển thị kết quả dự đoán
result_label = tk.Label(frame, text="Kết quả dự đoán sẽ được hiển thị ở đây.", font=('Helvetica', 14), fg='green', bg='lightblue')
result_label.grid(row=0, column=1, padx=20)  # Đặt kết quả bên phải

# Chạy vòng lặp sự kiện của giao diện người dùng
root.mainloop()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision.models as models

# Tải mô hình
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5 lớp cho 5 loại động vật
model.load_state_dict(torch.load('animal_classifier.pth'))
model.eval()

# Thiết lập các phép biến đổi cho hình ảnh
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_data_path = r'D:/animal-identification-main/animal-identification-main/data/animals/train'  # Đổi đường dẫn nếu cần
test_dataset = datasets.ImageFolder(test_data_path, transform=transform)  # Tạo dataset từ thư mục hình ảnh
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Tạo DataLoader để duyệt qua tập kiểm tra

class_names = ['capybara', 'chim-canh-cut', 'cho', 'ga', 'meo']  # Danh sách các lớp động vật

# Thu thập nhãn thực tế và nhãn dự đoán
y_true, y_pred = [], []
with torch.no_grad():  # Tắt tính toán gradient (dành cho đánh giá)
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Dự đoán từ mô hình
        _, predicted = torch.max(outputs, 1)  # Lấy nhãn có xác suất cao nhất
        y_true.extend(labels.cpu().numpy())  # Lưu nhãn thực tế
        y_pred.extend(predicted.cpu().numpy())  # Lưu nhãn dự đoán

# Tính toán và in ra các độ đo hiệu suất
accuracy = accuracy_score(y_true, y_pred)  # Tính độ chính xác
print(f"Accuracy: {accuracy * 100:.2f}%")  # In độ chính xác

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))  # Báo cáo phân loại

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)  # In ma trận nhầm lẫn



