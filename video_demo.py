from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def arr(x):

    
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    #----------------------------------------------------------Get x and y
    commaIndex_X1 = str(c1[0]).index(',')
    x1 = str(c1[0])[7:commaIndex_X1]
    x1 = int(x1)

    commaIndex_Y1 = str(c1[1]).index(',')
    y1 = str(c1[1])[7:commaIndex_Y1]
    y1 = int(y1)
    #------------------------
    commaIndex_X2 = str(c2[0]).index(',')
    x2 = str(c2[0])[7:commaIndex_X2]
    x2 = int(x2)

    commaIndex_Y2 = str(c2[1]).index(',')
    y2 = str(c2[1])[7:commaIndex_Y2]
    y2 = int(y2)

    x =x1,x2
    y = y1,y2
    obj = x,y
    return obj

def write(x, img):

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    #----------------------------------------------------------Get x and y
    commaIndex_X1 = str(c1[0]).index(',')
    x1 = str(c1[0])[7:commaIndex_X1]
    x1 = int(x1)

    commaIndex_Y1 = str(c1[1]).index(',')
    y1 = str(c1[1])[7:commaIndex_Y1]
    y1 = int(y1)
    #------------------------
    commaIndex_X2 = str(c2[0]).index(',')
    x2 = str(c2[0])[7:commaIndex_X2]
    x2 = int(x2)

    commaIndex_Y2 = str(c2[1]).index(',')
    y2 = str(c2[1])[7:commaIndex_Y2]
    y2 = int(y2)

    #print(x1,y1)
    #print(x2,y2)


    centerX = int(((x2-x1)/2) + x1)
    centerY = int(((y2-y1)/2)  + y1)
    #cv2.circle(img,(centerX, centerY), int(10), (0,255,0), -1) #draw center of the circle

    #----------------------------------------------------------
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()

def getLargestObj(objs):
    bigObj = objs[0]
    largest = 0
    for obj in objs:
        x1 = obj[0][0]
        x2 = obj[0][1]
        y1 = obj[1][0]
        y2 = obj[1][1]
        area= ((y2 - y1) * (x2 - x1) )
        if area > largest:
            largest = area
            bigObj = obj
    return bigObj

def drawClosestObj(obj, img):
    x1 = obj[0][0]
    x2 = obj[0][1]
    y1 = obj[1][0]
    y2 = obj[1][1]

    centerX = int(((x2-x1)/2) + x1)
    centerY = int(((y2-y1)/2)  + y1)
    cv2.circle(img,(centerX, centerY), int(20), (0,0,255), -1) #draw center of the circle
    return img


def displayObjInfo(objs):
    count = 0
    print("--------------------------------------------------")
    print("Number of cubes: " + str(len(objs)))
    for obj in objs:
        count+=1
        print("Cube #" + str(count))

        x1 = obj[0][0]
        x2 = obj[0][1]
        y1 = obj[1][0]
        y2 = obj[1][1]
        print("Point 1 : " + str(x1) + "," + str(y1))
        print("Point 2 :  " + str(y2) + "," + str(y2)) 
        print("Area : " + str( (y2 - y1) * (x2 - x1) ))   # (Y2- y1) * (x2 - x1)


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
	
    CUDA = torch.cuda.is_available()

    num_classes = 1

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    #save clip
    FILE_OUTPUT = 'saved.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (int(width), int(height)))



    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            

            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            

            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            #draw boxes
            list(map(lambda x: write(x, orig_im), output))

            #get array of objects bounding boxes
            objBoundBoxes = list(map(lambda x: arr(x), output)) 

            drawClosestObj(getLargestObj(objBoundBoxes), orig_im)

            cv2.imshow("frame", orig_im)


            out.write(orig_im)



            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("Saved video!")
                out.release()
                break
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
    

    
    

