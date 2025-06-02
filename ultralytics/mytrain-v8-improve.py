import sys
sys.path.append("/root/autodl-tmp/yolov8-new/ultralytics")  # 强制文件从这个路径下进行扫描
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__=='__main__':
    # Load a model
    model = YOLO(r'D:/yolov8/ultralytics/yolov8.yaml') # 不使用预训练权重训练
    # model = YOLO(r'/root/autodl-tmp/1-yolov8/yolov8/ultralytics/road-damage-datasets/yolov8.yaml').load("/root/autodl-tmp/1-yolov8/yolov8/ultralytics/yolov8n.pt") # 使用预训练权重训练
    # Train settings -------------------------------------------------------------------------------------------------------
    model.train(
        data=r'D:/yolov8/ultralytics/cfg/datasets/coco128.yaml', # path to data file, i.e. coco128.yaml
        epochs= 150 , # number of epochs to train for
        patience= 100 , # epochs to wait for no observable improvement for early stopping of training
        batch= 1 , # number of images per batch (-1 for AutoBatch)
        imgsz= 640 , # size of input images as integer or w,h
        save=True , # save train checkpoints and predict results
        save_period=-1 , # Save checkpoint every x epochs (disabled if < 1)
        cache=False , # True/ram, disk or False. Use cache for data loading
        device= 0 , # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers= 8 , # number of worker threads for data loading (per RANK if DDP)
        project= 'runs/train' , # project name
        name= 'RT-DETR-l' , # experiment name, results saved to 'project/name' directory
        exist_ok= False , # whether to overwrite existing experiment
        pretrained= True , # whether to use a pretrained model
        optimizer= 'SGD' , # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
        verbose= True , # whether to print verbose output
        seed= 0 , # random seed for reproducibility
        deterministic= True , # whether to enable deterministic mode
        single_cls= False , # train multi-class data as single-class
        rect= False , # rectangular training if mode='train' or rectangular validation if mode='val'
        cos_lr= False , # use cosine learning rate scheduler
        close_mosaic= 0 , # (int) disable mosaic augmentation for final epochs
        resume= True , # resume training from last checkpoint
        amp= True , # Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
        # Segmentation
        overlap_mask= True , # masks should overlap during training (segment train only)
        mask_ratio= 4 , # mask downsample ratio (segment train only)
        # Classification
        dropout= 0.0 , # use dropout regularization (classify train only)
        
        # Hyperparameters ------------------------------------------------------------------------------------------------------
        lr0= 0.01 , # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf= 0.01 , # final learning rate (lr0 * lrf)
        momentum= 0.937 , # SGD momentum/Adam beta1
        weight_decay= 0.0005 , # optimizer weight decay 5e-4
        warmup_epochs= 3.0 , # warmup epochs (fractions ok)
        warmup_momentum= 0.8 , # warmup initial momentum
        warmup_bias_lr= 0.1 , # warmup initial bias lr
        box= 7.5 , # box loss gain
        cls= 0.5 , # cls loss gain (scale with pixels)
        dfl= 1.5 , # dfl loss gain
        pose= 12.0 , # pose loss gain
        kobj= 1.0 , # keypoint obj loss gain
        label_smoothing= 0.0 , # label smoothing (fraction)
        nbs= 64 , # nominal batch size
        hsv_h= 0.015 , # image HSV-Hue augmentation (fraction)
        hsv_s= 0.7 , # image HSV-Saturation augmentation (fraction)
        hsv_v= 0.4 , # image HSV-Value augmentation (fraction)
        degrees= 0.0 , # image rotation (+/- deg)
        translate= 0.1 , # image translation (+/- fraction)
        scale= 0.5 , # image scale (+/- gain)
        shear= 0.0 , # image shear (+/- deg)
        perspective= 0.0 , # image perspective (+/- fraction), range 0-0.001
        flipud= 0.0 , # image flip up-down (probability)
        fliplr= 0.5 , # image flip left-right (probability)
        mosaic= 1.0 , # image mosaic (probability)
        mixup= 0.0 , # image mixup (probability)
        copy_paste= 0.0 , # segment copy-paste (probability)
                )
    