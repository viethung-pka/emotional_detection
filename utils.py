
import cv2
import mediapipe as mp
import math
import numpy as np

LANDMARKS_EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334]

LANDMARKS_EYES = [33, 160, 158, 133, 153, 144, 362, 387, 385, 263, 373, 380]

LANDMARKS_MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 
    13, 82, 81, 42, 183, 78, 95
]
LANDMARKS_CHEEKS = [229, 31, 230, 231, 117, 118, 119, 120, 50, 205, 36, 142, 100, 101, 451, 450, 449, 448, 346, 347, 348, 349, 329, 371, 226, 425, 280, 330]

LANDMARKS_NOSE_CHEEKS = [1, 2, 98, 327, 94, 331, 78, 82, 13, 312]

LANDMARKS_FACE_CONTOUR = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 
    103, 67, 109
]

IMPORTANT_LANDMARKS = (
    LANDMARKS_EYEBROWS + LANDMARKS_EYES + LANDMARKS_MOUTH +
    LANDMARKS_NOSE_CHEEKS + LANDMARKS_CHEEKS
)

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    #   + (p1.z - p2.z)**2)

def get_face_landmarks(image, face_mesh, draw=False):

    """
    Lấy các khoảng cách Euclid giữa các điểm landmarks.
    """

    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_input_rgb)

    distances = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        for i in range(len(IMPORTANT_LANDMARKS) - 1):
            for j in range(i + 1, len(IMPORTANT_LANDMARKS)):
                idx1 = IMPORTANT_LANDMARKS[i]
                idx2 = IMPORTANT_LANDMARKS[j]
                p1 = face_landmarks[idx1]
                p2 = face_landmarks[idx2]
                distance = euclidean_distance(p1, p2)
                distances.append(distance)

    else:
        print("Không tìm thấy khuôn mặt.")
        return None

    return distances