import cv2
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_name="buffalo_l", device=-1):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=device)

    def detect(self, image_path):
        img = cv2.imread(image_path)
        faces = self.app.get(img)

        return img, faces
