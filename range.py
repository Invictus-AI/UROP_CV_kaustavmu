import os
import time
import csv

#format: sizes = [[imgsize, batchsize, epochs], i.e. [512, 2, 3], [256, 4, 2], etc.],
sizes = []
#format: strings with the yaml file name without extension. Each dataset should have a separate yaml file.
yamlfiles = []

#Don't touch anything under here!

vals = [['imgsize', 'batchsize', 'totalepochs', 'yamlfile', 'timetaken', 'epoch', 'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']]

for i in sizes:
    for j in yamlfiles:
        seconds = time.time()
        os.system("python train.py --img " + str(i[0]) + " --batch " + str(i[1]) + " --epochs " + str(i[2]) + " --data ./data/" + j + ".yaml --weights ./weights/yolov5x.pt")
        newtime = time.time() - seconds
        dirlist = os.listdir('./runs/train')
        lastpath = './runs/train/' + sorted(dirlist, key=lambda x: int(x[3:]+"0"))[-1] + "/results.csv"
        file = open(lastpath)
        csvreader = csv.reader(file)
        vals = vals + [[i[0], i[1], i[2], j, newtime] + row for row in csvreader][1:]

#You may edit the path of the results file below:

with open('resultsfile.csv', 'w', newline = '') as file:
    mywriter = csv.writer(file, delimiter = ',')
    mywriter.writerows(vals)
