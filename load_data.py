import os
import cv2
import numpy as np
import random
import mediapipe as mp
from utils import get_face_landmarks

data_dir = './emotion_new/images/train'

output_file = 'emotion_xy_.txt'

output = []

mp_face_mesh = mp.solutions.face_mesh

emotion_labels = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'surprise': 3  
}

def add_padding(image, top, bottom, left, right, color=(0, 0, 0)):
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

for emotion_folder in os.listdir(data_dir):
    emotion_folder_path = os.path.join(data_dir, emotion_folder)

    if os.path.isdir(emotion_folder_path):
        emotion_label = emotion_labels.get(emotion_folder, -1)  

        if emotion_label == -1:
            print(f"Không tìm thấy nhãn cho thư mục {emotion_folder}, bỏ qua.")
            continue

        print(f"Đang xử lý cảm xúc: {emotion_folder} (label: {emotion_label})")

        image_paths = os.listdir(emotion_folder_path)
        num_images = len(image_paths)

        selected_images = random.sample(image_paths, num_images)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            for image_name in selected_images:
                image_path = os.path.join(emotion_folder_path, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Lỗi khi đọc ảnh: {image_name}, bỏ qua.")
                    continue

                # Xử lý ảnh gốc
                distances = get_face_landmarks(image, face_mesh)
                if distances:
                    all_distances = []
                    all_distances.extend(distances)
                    all_distances.append(emotion_label)
                    output.append(all_distances)
                    print(f"Xử lý thành công ảnh gốc: {image_name}, số lượng khoảng cách: {len(all_distances) - 1}")
                else:
                    print(f"Lỗi khi xử lý ảnh gốc: {image_name}")

                # Thêm padding cho ảnh với các giá trị khác nhau
                paddings = [(50, 0, 50, 0), (0, 50, 0, 50), (20, 20, 20, 20)]  # padding (top, bottom, left, right)
                for pad in paddings:
                    padded_image = add_padding(image, *pad)

                    # Lấy các điểm landmarks từ ảnh đã thêm padding
                    distances = get_face_landmarks(padded_image, face_mesh)

                    if distances:
                        all_distances = []
                        all_distances.extend(distances)
                        all_distances.append(emotion_label)
                        output.append(all_distances)

                        print(f"Xử lý thành công ảnh: {image_name} với padding {pad}, số lượng khoảng cách: {len(all_distances) - 1}")
                    else:
                        print(f"Lỗi khi xử lý ảnh: {image_name} với padding {pad}")

# Lưu mảng output vào file
output_array = np.asarray(output)
np.savetxt(output_file, output_array, fmt='%f')

print(f"Lưu dữ liệu thành công vào {output_file}!")
