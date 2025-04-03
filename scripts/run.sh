python  -m torch.distributed.run \
        --nproc_per_node 8 \
        --master_port 11527 \
        train_dual.py \
        --workers 8 \
        --device 0,1,2,3,4,5,6,7 \
        --sync-bn \
        --batch-size 128 \
        --data data/coco.yaml \
        --img 640 \
        --cfg models/detect/s-shuffle-40x40-split.yaml \
        --weights '' \
        --name s-shuffle-40x40-k30 \
        --hyp hyp.scratch-low.yaml \
        --epochs 1

python  train_dual.py \
        --workers 8 \
        --device 0 \
        --batch 1 \
        --data data/subCoco.yaml \
        --img 640 \
        --cfg models/detect/s-shuffle-40x40-split.yaml \
        --weights '' \
        --name test_shape \
        --hyp hyp.scratch-low.yaml \
        --epochs 1

python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port 11527 \
        train_dual.py \
        --workers 8 \
        --device 0,1,2,3,4,5,6,7 \
        --sync-bn \
        --batch 128 \
        --data data/coco.yaml \
        --img 640 \
        --cfg models/detect/yolov9-t.yaml \
        --weights '' \
        --name yolov9-t \
        --hyp hyp.scratch-low.yaml \
        --min-items 0 \
        --epochs 500 \
        --close-mosaic 10


python  val.py \
        --data data/subCoco.yaml \
        --img 640 \
        --batch 16 \
        --conf 0.001 \
        --iou 0.7 \
        --device 6 \
        --weights '/home2/zhouzhixiang/Hyper-YOLOv1.1/runs/train/yolov9-s-hyper-baseline/weights/best-converted.pt' \
        --save-json \
        --name hyper-s-baseline

python  val_dual.py \
        --data data/coco.yaml \
        --img 640 \
        --batch 16 \
        --conf 0.001 \
        --iou 0.7 \
        --device 6 \
        --weights '/home/lisiqi/zhouzhixiang/Hyper-YOLOv1.1/runs/train/yolov9-s-hyper-baseline2/weights/best.pt' \
        --save-json \
        --name baseline

python  detect.py \
        --source '/home2/zhouzhixiang/Hyper-YOLOv1.1/data/images/000000000009.jpg' \
        --img 640 \
        --device 6 \
        --weights '/home2/zhouzhixiang/Hyper-YOLOv1.1/runs/train/yolov9-s-hyper-baseline/weights/best-converted.pt' \
        --name hyper-s-baseline

python  detect_dual.py \
        --source '/home/lisiqi/zhouzhixiang/Hyper-YOLOv1.1/data/images/000000000009.jpg' \
        --img 640 \
        --device 1 \
        --weights '/home/lisiqi/zhouzhixiang/Hyper-YOLOv1.1/runs/train/yolov9-s-hyper-baseline2/weights/best.pt' \
        --name test_hg