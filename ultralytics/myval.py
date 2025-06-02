import sys
sys.path.append("/root/autodl-tmp/yolov8-new/ultralytics")  # 强制文件从这个路径下进行扫描
from ultralytics import YOLO

if __name__== '__main__':
    # 加载模型
    model = YOLO(r'D:/yolov8/weights/best.pt')  #
    # Val/Test settings
    model.val(
        val=True,  # validate/test during training
        data=r'D:/yolov8/ultralytics/data.yaml',
        split='test',  # (int)dataset split to use for validation, i.e. 'val', 'test' or 'train'
        batch=1,  # (int) number of images per batch (-1 for AutoBatch)
        imgsz=640,  # (int) size of input images or w,h
        workers=8,  # (int) number of worker threads for bata loading (per PANK if DDP)
        save_json=False,  # save results to JSON file
        save_hybrid=False,  # save hybrid version of labels (labels + additional predictions)
        conf=0.001,  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.6,  # intersection over union (IoU) threshold for NMS
        project='runs/val',  # (str,optional)project name
        name='300轮-yolov8新版本-1th-RDD2022-all-改进模型-测',  # (str,optional)project name, results saved to 'project/name' directory
        max_det=300,  # maximum number of detections per image
        half=False,  # use half precision (FP16)
        dnn=False,  # use OpenCV DNN for ONNX inference
        plots=True,  # save plots during train/val
    )
results = model('D:/yolov8/images/248.jpg', save=True)  # predict on an image
#yolo detect predict model=D:/yolov8/weights/best.pt source='D:/yolov8/images/123.jpg'