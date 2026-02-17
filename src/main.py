import os
import cv2
from detector import FaceDetector
import numpy as np
from sklearn.cluster import DBSCAN
import shutil

IMAGE_FOLDER = "../data/input-frames/"
CROP_FOLDER = "../data/output-detected-faces"
TARGET_SIZE = (256,256)
OUTPUT_FOLDER = "../data/groups"

detector = FaceDetector(device=-1)

embeddings = []
image_references = []

for file in os.listdir(IMAGE_FOLDER):
    path = os.path.join(IMAGE_FOLDER, file)

    img, faces = detector.detect(path)

    print(f"{file} → {len(faces)} faces detected")
   
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox


        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
        image_references.append((file,i))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        face_crop = img[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop,TARGET_SIZE)

        crop_filename = f"{os.path.splitext(file)[0]}_face{i}.jpg"
        save_path = os.path.join(CROP_FOLDER, crop_filename)

        cv2.imwrite(save_path, face_crop)

clustering = DBSCAN(
eps=0.5,            # similarity threshold
min_samples=2,      # minimum faces to form group
metric='cosine')
labels = clustering.fit_predict(embeddings)
print(labels)
for label, (filename, face_index) in zip(labels, image_references):

    if label == -1:
        continue  # skip noise

    person_folder = os.path.join(OUTPUT_FOLDER, f"person_{label}")
    os.makedirs(person_folder, exist_ok=True)

    crop_name = f"{os.path.splitext(filename)[0]}_face{face_index}.jpg"
    src_path = os.path.join(CROP_FOLDER, crop_name)


    # To avoid overwriting
    new_name = f"{os.path.splitext(filename)[0]}_face{face_index}.jpg"
    dst_path = os.path.join(person_folder, new_name)

    shutil.copy(src_path, dst_path)
cv2.destroyAllWindows()
