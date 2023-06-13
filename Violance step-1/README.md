## Obyektlarni aniqlash, segmentatsiya qilish va kuzatish dasturi

![image](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/image.jpg)

### Ushbu dasturda kuzatilishi kerak bo'lgan eng asosiy 2 ta obyekt mavjud:
#### 1. Piyoda (padestrian)
#### 2. Avtoulov (vehicle)


### Dastur 3 ta asosiy funksiyani bajaradi:

#### 1. Obyektlarni aniqlash (Object detection)

[![detect](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/detection.png)](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/detection.mp4)

```shell
C:\User\violance> python deploy.py --src video.mp4 --yolo v8m --mode detection
```

#### 2. Obyektlarni segmentatsiyalash (Object segmentation)

[![segment](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/segmentation.png)](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/segmentation.mp4)

```shell
C:\User\violance> python deploy.py --src video.mp4 --yolo v8m --mode segmentation
```

#### 3. Obyektlarni kuzatish (Object tracking)

[![track](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/tracking.png)](https://github.com/MisterFoziljon/AIRI-Violance/blob/main/Violance%20step-1/demonstration/tracking.mp4)

```shell
C:\User\violance> python deploy.py --src video.mp4 --yolo v8m --mode tracking
```


