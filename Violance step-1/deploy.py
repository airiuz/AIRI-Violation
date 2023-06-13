from ultralytics import YOLO
import numpy as np
import torch
import cv2
import argparse


def find_padestrian(people,vehicle,width,height):
    
    padestrian = []

    for person in people:
        
        perBox = person[0]
        in_car = False
        
        for car in vehicle:
            carBox = car[0]
            
            in_car = in_car or max(carBox[0]-10,0) <= perBox[0] and perBox[2] <= min(carBox[2] + 10,width) and max(carBox[1]-10,0) <= perBox[1] and perBox[3] <= min(carBox[3] + 10,height)
            
            if in_car:
                break
                
        if not in_car:
            padestrian.append(person)
                
    return padestrian


def detection(elements,cls,frame):
    
    for element in elements:
        
        (xmin,ymin,xmax,ymax),id,mask,color = element

        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 1)

        if int(cls)==0:
            cv2.putText(frame, "padestrian", (xmin, ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)
        else:
            cv2.putText(frame, "vehicle", (xmin, ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)


def segmentation(elements,cls,frame):
    
    for element in elements:
        
        (xmin,ymin,xmax,ymax),id,mask,color = element
        cv2.polylines(frame, [mask], True, color, 2) 

        if int(cls)==0:
            cv2.putText(frame, "padestrian", (xmin, ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)

        else:
            cv2.putText(frame, "vehicle", (xmin, ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)


def tracking(elements,frame):
    
    colors = [
        (255, 0, 123),(0, 255, 123),(123, 0, 255),(255, 123, 0),(0, 123, 255),
        (123, 255, 0),(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 255),
        (0, 0, 0),(255, 255, 0),(255, 0, 255),(0, 255, 255),(123, 123, 123),
        (100, 150, 200),(50, 75, 100),(200, 100, 50),(75, 100, 50),(100, 50, 75),
        (50, 100, 75),(150, 200, 100),(200, 150, 100),(75, 50, 100),(100, 75, 50),
        (150, 100, 200),(100, 200, 150),(200, 100, 150),(100, 150, 50),(150, 50, 100),
        (0, 100, 200),(200, 0, 100),(100, 0, 200),(200, 100, 0),(100, 200, 0),
        (0, 200, 100),(123, 0, 0),(0, 123, 0),(0, 0, 123),(123, 123, 0),
        (123, 0, 123),(0, 123, 123),(123, 123, 255),(255, 123, 123),(123, 255, 123),
        (123, 123, 50),(50, 123, 123),(123, 50, 123),(123, 200, 255),(255, 123, 200)]
    
    for element in elements:
        
        (xmin,ymin,xmax,ymax),id,mask,color = element

        cv2.polylines(frame, [mask], True, colors[int(id)-1], 2)
        cv2.putText(frame, f"id-{int(id)}", (xmin, ymin+20),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 2)


def main(args: argparse.Namespace) -> None:

    if args.yolo=="v8m":
        segmentation_model_path = 'models/yolov8m-seg.pt'
    else:
        segmentation_model_path = 'models/yolov8x-seg.pt'

    segmentation_model = YOLO(segmentation_model_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    video = cv2.VideoCapture(args.src)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

    if args.mode == "detection":
        filename = f'demonstration/detection.mp4'

    elif args.mode == "segmentation":
        filename = f'demonstration/segmentation.mp4'

    elif args.mode == "tracking":
        filename = f'demonstration/tracking.mp4'
        
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), fps-5, (width,height))

    epoch = 0
    
    while True:
        
        ret, frame = video.read()
        frame = cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)
        
        results = segmentation_model.track(frame,
                                           conf = 0.5,
                                           iou = 0.5,
                                           persist=True,
                                           retina_masks=True,
                                           device = device,
                                           classes = [0,2,3,5,7])
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id
        
        if ids is not None:
            
            person = []
            vehicle = []
            
            labels = results[0].boxes.cls.cpu().numpy().astype(int) 
            masks = results[0].masks.xy
            
            ids = ids.cpu().numpy().astype(int)%100
            
            for box, id, cls, mask in zip(boxes, ids, labels, masks):
                
                mask = np.array(mask).astype('int')
                mask = mask.reshape((-1, 1, 2))
                
                if int(cls)==0:
                    person.append([box,id,mask,(255,255,255)])

                else:
                    vehicle.append([box,id,mask,(129,25,255)])
                
            padestrian = find_padestrian(person, vehicle, width, height)

            if args.mode == "detection":
                if len(padestrian)>0:
                    detection(padestrian,0,frame)
                
                if len(vehicle)>0:
                    detection(vehicle,1,frame)
                    
            if args.mode == "segmentation":
                if len(padestrian)>0:
                    segmentation(padestrian,0,frame)
                
                if len(vehicle)>0:
                    segmentation(vehicle,1,frame)

            if args.mode == "tracking":
                if len(padestrian)>0:
                    tracking(padestrian,frame)
                
                if len(vehicle)>0:
                    tracking(vehicle,frame)
                    
        out.write(frame)
        cv2.imshow("video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    out.release()
    video.release()
    cv2.destroyAllWindows() 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='video file')
    parser.add_argument('--yolo', type=str, help='model')
    parser.add_argument('--mode', type=str, help='video file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
