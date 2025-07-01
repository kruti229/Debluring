import cv2
import os

video_path='D:/Create/freelance/deblurring/model/video127.mp4'
output_folder='frames'
os.makedirs(output_folder,exist_ok=True)

cap=cv2.VideoCapture(video_path)
frame_count=0

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print('got none')
        break
    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.png",frame)
    frame_count += 1

cap.release()
cv2.destroyWindow()