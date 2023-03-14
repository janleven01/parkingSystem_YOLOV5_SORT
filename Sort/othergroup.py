from main import stream as PSinfo
import cv2
import torch
import pandas as pd
model = torch.hub.load('', 'custom', 
        path= 'runs/train/exp/weights/best.pt', 
        source = 'local')
frame = cv2.VideoCapture('videos/brown minivan.mp4')

def main():
 while True:
   info = PSinfo(frame,model)
   info_List = list(info.values())   
   for i in range(len(info_List)):
        data = info_List[i][1][4]
        print(type(data))
if __name__ == "__main__":
 main()
 





