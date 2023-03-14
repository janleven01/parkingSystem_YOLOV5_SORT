import cv2
import numpy as np
import torch 
import os
from sort import Sort
from color_recognition_module import knn_classifier
from color_recognition_module import color_histogram_feature_extraction
from yolov5.utils.plots import colors
import schedule
import time
import pandas as pd

#Load to model
model = torch.hub.load('', 'custom', 
        path= 'runs/train/exp/weights/best.pt', 
        source = 'local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', 
#         path= 'runs/train/exp10/weights/best.pt', 
#         force_reload = True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
#         pretrained = True)

cap = cv2.VideoCapture('videos/Test1.mp4')

savepath = os.path.join(os.getcwd(), 'data', 'video')
mot_tracker = Sort()

class_id = {'Pickup':0,'Hatchback':1,'Sedan':2,'SUV':3,'Minivan':4,'Wagon':5,'Others':6}

#Coordinates for parking space
W_region = [110,407],[3,631],[54,1070],[1436,900],[1930,676],[1920,413],[1727,288],[110,407]
A1 = [587,617],[384,776],[533,867],[729,676]
A2 = [821,594],[643,767],[789,832],[958,643]
A3 = [1030,565],[883,736],[1042,791],[1167,607]
A4 = [1225,539],[1107,703],[1287,767],[1372,579]
A5 = [1401,517],[1332,680],[1497,727],[1528,555]
A6 = [1544,499],[1521,628],[1648,672],[1655,535]
A7 = [1662,486],[1658,602],[1761,640],[1759,517]
B1 = [639,454],[474,559],[588,606],[750,487]
B2 = [818,437],[686,538],[821,583],[948,470]
B3 = [1001,422],[899,515],[1028,555],[1126,453]
B4 = [1174,406],[1093,493],[1225,529],[1298,437]
B5 = [1332,398],[1274,471],[1399,510],[1439,427]
B6 = [1457,393],[1428,457],[1541,490],[1555,419]    
B7 = [1561,384],[1554,445],[1658,473],[1655,409]

#Parking space Initialiation
row_Array = ['A1','A2','A3','A4','A5','A6','A7','B1','B2','B3','B4','B5','B6','B7']

row_A_array = [A1,A2,A3,A4,A5,A6,A7]
posList_A = []

for i in range(len(row_A_array)):        
    posList_A.append(row_A_array[i])
    
row_B_array = [B1,B2,B3,B4,B5,B6,B7]
posList_B = []
for i in range(len(row_B_array)):
    posList_B.append(row_B_array[i])

parked_car_A = {}
parked_car_B = {}
row_dict = {}
class_id = {'Pickup':(0,0,255),'Hatchback':(0,255,255),'Sedan':(0,255,255),'SUV':(0,255,255),'Minivan':(0,255,0),'Wagon':(0,255,0),'Others':(0,255,0)}

#Color Initialization
detected_IDs = {}
prediction = ''

#Clear color_detected dictionary every 30 mins
def clearDict():
    detected_IDs.clear()
schedule.every(30).minutes.do(clearDict)


#Color Detection Function 
def checkColor(obj_id,frame,x1,y1,x2,y2):
    # Check if object's ID is not yet detected
    if obj_id not in detected_IDs :
        cropped_img = frame[y1:y2,x1:x2]  
        cv2.imwrite('car.jpg',cropped_img)
        color_histogram_feature_extraction.color_histogram_of_test_image('car.jpg') 
        prediction = knn_classifier.main('training.data', 'test.data')
        new_dict = {obj_id:prediction}
        detected_IDs.update(new_dict)
    # else:
    #     cv2.putText(frame,"Color: " + detected_IDs.get(obj_id),(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255), 1)
    #     cv2.putText(frame,"Color: " + detected_IDs.get(obj_id),(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255), 2)

# Check Parking Space in row_A Function
def checkPS_row_A(x1,y1,x2,y2,cx,cy,frame,obj_id,class_name):
    # Loop each coordinates in row_A 
    for i, area in enumerate(posList_A):
        # Check if the midpoint of the detected object hit the line or enters the polygon
        check_firstPoint = cv2.pointPolygonTest(np.array(area),(cx,cy), True)
        if check_firstPoint >= 0:
            # Check if the bottom left of the bounding box hit or enters the polygon
            check_secondPoint = cv2.pointPolygonTest(np.array(area),(x1,y2), True)
            if check_secondPoint >= 0:
                parked_car_A[i] = area
                row = [i+1,area,'Occupied',obj_id,class_name,detected_IDs.get(obj_id),[x1,y1,x2,y2]]
                row_dict[row_Array[i]] = row
                color = (0,0,255)
                checkColor(obj_id,frame,x1,y1,x2,y2)
            else:
                color = (0,255,0)
                parked_car_A.pop(i,None)
                parked_car_A.pop(None,None)
                row_dict[row_Array[i]] = [i+1,[0,0,0,0],'Unoccupied',0,0,0,[0,0,0,0]]
          #  cv2.polylines(frame,[np.array(area,np.int32)],True,(color),6)
        if not area in parked_car_A.values():
          #  cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),6) 
            row_dict[row_Array[i]] = [i+1,[0,0,0,0],'Unoccupied',0,0,0,[0,0,0,0]]
        
def checkPS_row_B(x1,y1,x2,y2,cx,cy,frame,obj_id,class_name):
    # Loop each coordinates in row_B 
    for i, area in enumerate(posList_B):
        # Check if the midpoint of the detected object hit or enters the polygon
        check_firstPoint = cv2.pointPolygonTest(np.array(area),(x1+30,y2-30), True)     
        #cv2.circle(frame,(x1+50,cy+20),10,(0,255,255),-1)Q
        if check_firstPoint >= 0:
            # Check if the bottom left of the bounding box hit the line or enters the polygon
            check_secondPoint = cv2.pointPolygonTest(np.array(area),(cx,cy), True)
            if check_secondPoint >= 0:   
                parked_car_B[i] = area
                row = [i+8,area,'Occupied',obj_id,class_name,detected_IDs.get(obj_id),[x1,y1,x2,y2]]
                row_dict[row_Array[i+7]] = row
                color = (0,0,255)
                checkColor(obj_id,frame,x1,y1,x2,y2)
            else:
                parked_car_B.pop(i,None)
                parked_car_B.pop(None,None)
                row_dict[row_Array[i+7]] = [i+8,[0,0,0,0],'Unoccupied',0,0,0,[0,0,0,0]]
                color = (0,255,0)   
          #  cv2.polylines(frame,[np.array(area,np.int32)],True,(color),3)
        elif not area in parked_car_B.values():
          #  cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),3)
            row_dict[row_Array[i+7]] = [i+8,[0,0,0,0],'Unoccupied',0,0,0,[0,0,0,0]]

def stream(cap,model):

    while(True):
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 640))
        preds = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # preds = model(frame)
        df = preds.pandas().xyxy[0]
        to_sort = df.to_numpy()
        track_bbs_ids = mot_tracker.update(to_sort)
        for j, track in enumerate(track_bbs_ids):
            coords = track_bbs_ids.tolist()[j]       
            x1, y1, x2, y2 = int(float(coords[0])), int(float(coords[1])), int(float(coords[2])), int(float(coords[3]))
            obj_id = int(coords[4])
            class_name = str(coords[5])
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            inside_W_region = cv2.pointPolygonTest(np.array(W_region),(x2,y1), True)    
            if inside_W_region >= 0:
                ID = str(obj_id)
                if class_name in class_id:
                    color_cls = class_id.get(class_name)
                label = class_name + ': ' + ID
                color = color_cls
                checkPS_row_A(x1,y1,x2,y2,cx,cy,frame,obj_id,class_name)
                checkPS_row_B(x1,y1,x2,y2,cx,cy,frame,obj_id,class_name)
                #1920p
                cv2.rectangle(frame,(x1,y1),(x1+125,y1-30),color,-1)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color, 2)
                cv2.putText(frame,label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255), 2)
                #640p
                # cv2.rectangle(frame,(x1,y1),(x1+60,y1-20),color,-1)
                # cv2.rectangle(frame,(x1,y1),(x2,y2),color, 1)
                # cv2.putText(frame,label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255), 1)
                # cv2.circle(frame,(x1+30,y2-30),10,(0,255,255),-1)
      
        cv2.rectangle(frame,(3,53),(350,140),(0,255,0), -1)
        parked_car = 14 - (int(len(parked_car_A)) + int(len(parked_car_B))) 
        cv2.putText(frame,f'Vacant: {parked_car}',(5,120), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0), 5)
        schedule.run_pending()
        cv2.namedWindow('frame', 0)
        cv2.resizeWindow('frame', 1200, 600)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
           break   

        #comment if you want
        #return row_dict      



if __name__ == "__main__":
    stream(cap,model)
    cv2.destroyAllWindows()
    