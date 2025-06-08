import cv2
import base64
import numpy as np
from retinaface import RetinaFace
from config.settings import Config
import os

def crop_and_save_face(image_data, output_path, filename):
    try:
        detector = RetinaFace
        
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_data

        faces = detector.detect_faces(image)

        if isinstance(faces, dict):
            face_data = faces['face_1']
            facial_area = face_data['facial_area']
            
            x = facial_area[0]
            y = facial_area[1]
            width = facial_area[2] - facial_area[0]
            height = facial_area[3] - facial_area[1]
            
            margin = 20
            x = max(int(x - margin), 0)
            y = max(int(y - margin), 0)
            width = int(width + margin * 2)
            height = int(height + margin * 2)

            h, w = image.shape[:2]
            width = min(width, w - x)
            height = min(height, h - y)

            cropped_face = image[y:y+height, x:x+width]
            cropped_face_resized = cv2.resize(cropped_face, (250, 250))

            output_file = os.path.join(output_path, filename)
            cv2.imwrite(output_file, cropped_face_resized)
            return filename
        else:
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            x = max(center_x - 125, 0)
            y = max(center_y - 125, 0)
            
            cropped_face = image[y:y+250, x:x+250]
            cropped_face_resized = cv2.resize(cropped_face, (250, 250))

            output_file = os.path.join(output_path, f'center_crop_{filename}')
            cv2.imwrite(output_file, cropped_face_resized)
            return f'center_crop_{filename}'
    except Exception as e:
        print(f"Error processing image: {e}")
        return None