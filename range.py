import os
import time
import csv

sizes = []
times = []
vals = [['epoch', 'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']]

for i in sizes:
    seconds = time.time()
    os.system("python train.py --img " + i[0] + " --batch " + i[1] + " --epochs " + i[2] + " --data ./data/coco.yaml --weights ./weights/yolov5x.pt")
    times.append(time.time()-seconds)
    dirlist = os.listdir('./runs/train')
    lastpath = './runs/train/' + sorted(dirlist, key=lambda x: int(x[3:]+"0"))[-1] + "/results.csv"
    file = open(lastpath)
    csvreader = csv.reader(file)
    vals.append([row for row in csvreader][-1])

print(sizes)
print(times)
print(vals)
