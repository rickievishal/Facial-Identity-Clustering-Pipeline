import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

print("InsightFace loaded successfully")
