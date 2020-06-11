import os
from cv2 import cv2
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from imageai.Detection import VideoObjectDetection

video = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(video, output_file_path=r"C:", frames_per_second=24, log_progress=True)

"""while True:
    _, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()"""