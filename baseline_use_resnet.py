import os
import sys
import cv2
import resnet
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
from skimage import transform
from torch.autograd import Variable
from torchvision import datasets,models,transforms
from PIL import Image,ImageDraw
from torchvision import models
from pyhog import pyhog

outputlayer = [1,2,3,4]
numlayers = len(outputlayer)
layerweights = [0.25,0.5,1.2,1]

lam = 1e-4
cell_size = 4

assert (numlayers == len(layerweights))

# network init

model = resnet.ResNet(layers = [2,2,2,2],outlayers=outputlayer)

# load partial weights
model_dict = model.state_dict()

# absolute path
#params = torch.load('resnet34-333f7ec4.pth')
params = torch.load('resnet18-5c106cde.pth')
load_dict = {k:v for k,v in params.items() if 'fc' not in k}
model_dict.update(load_dict)
model.load_state_dict(model_dict)
model.cuda()



def get_search_windows(size,im_size):

    if(size[0] / size[1] > 2):
        #For object with large height
        window_size = np.floor(np.multiply(size, [1+0.4,1+1.8]))

    elif np.prod(size)/np.prod(im_size) > 0.05:
        window_size = np.floor(size * (1 + 1))

    else:
        window_size = np.floor(size * (1 + 1.8))

    return window_size


def get_optical_flow(pre_image,next_image):

    pre_image = cv2.cvtColor(pre_image,cv2.COLOR_BGR2GRAY)

    next_image = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)

    assert(pre_image.shape == next_image.shape)

    feaPts = np.array([pre_image.shape[1] / 2, pre_image.shape[0] / 2], dtype=np.float32)
    feaPts = feaPts.reshape(-1, 1, 2)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    new_position, status, trackError = cv2.calcOpticalFlowPyrLK(pre_image, next_image, feaPts, None, **lk_params)

    if new_position is not None:

        newPts = new_position[status == 1]

        if len(newPts) != 0 :

            w, h = [float(newPts[0][0] - feaPts[0][0][0]), float(newPts[0][1] - feaPts[0][0][1])]

            if np.fabs(w) < 10 and np.fabs(h) < 10:

                return w,h

    return 0,0

def get_ori(image,position,wsz,scale_factor = None):

    sz_ori = wsz

    patch_wsz = wsz
    if scale_factor != None:
        patch_wsz = np.floor(patch_wsz*scale_factor)

    y = np.floor(position[0])-np.floor(patch_wsz[0]/2) + np.arange(patch_wsz[0], dtype=int)
    x = np.floor(position[1])-np.floor(patch_wsz[1]/2) + np.arange(patch_wsz[1], dtype=int)

    x,y = x.astype(int),y.astype(int)

    #check bounds
    x[x < 0] = 0
    y[y < 0] = 0

    x[x >= image.shape[1]] = image.shape[1] - 1
    y[y >= image.shape[0]] = image.shape[0] - 1

    ori = image[np.ix_(y,x)]

    if scale_factor != None:
        ori = misc.imresize(ori,sz_ori.astype(int))

    return ori

def pre_process_image(ori):

    imgMean = np.array([0.485, 0.456, 0.406], np.float)
    imgStd = np.array([0.229, 0.224, 0.225])
    ori = transform.resize(ori, (224, 224))
    ori = (ori - imgMean) / imgStd
    ori = np.transpose(ori, (2, 0, 1))
    ori = torch.from_numpy(ori[None, :, :, :]).float()
    ori = Variable(ori)
    if torch.cuda.is_available():
        ori = ori.cuda()
    return ori

def get_feature(ori):

    feature_ensemble = model(ori)

    return feature_ensemble

def get_scale_window(image,position,target_size,sfs,scale_window,scale_model_sz):

    #pos = [position[1],position[0]]
    #ts = np.array([target_size[1],target_size[0]])

    out = []
    for i in range(len(sfs)):
        patch_sz = np.floor(target_size * sfs[i])
        scale_patch = get_ori(image,position,patch_sz)
        im_patch_resized = transform.resize(scale_patch,scale_model_sz,mode='reflect')
        temp_hog = pyhog.features_pedro(im_patch_resized, 4)
        out.append(np.multiply(temp_hog.flatten(),scale_window[i]))

    return np.asarray(out)

def get_filter(feature_ensemble,yf,cos_window):

    num = []
    den = []

    for i in range(numlayers):

        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))

        x = ndimage.zoom(feature,(float(cos_window.shape[0])/feature.shape[0],float(cos_window.shape[1])/feature.shape[1],1),order=1)

        x = np.multiply(x,cos_window[:,:,None])
        xf = np.fft.fft2(x,axes=(0,1))

        num.append(np.multiply(yf[:,:,None], np.conj(xf)))
        den.append(np.real(np.sum(np.multiply(xf,np.conj(xf)),axis=2)))

    return num,den

def get_scale_filter(image,position,target_size,current_scale_factor,scaleFactors,scale_window,model_size,ysf):

    sw = get_scale_window(image,position,target_size,current_scale_factor*scaleFactors,scale_window,model_size)
    swf = np.fft.fftn(sw,axes=[0])
    s_num = np.multiply(ysf[:,None],np.conj(swf))
    s_den = np.real(np.sum(np.multiply(swf,np.conj(swf)), axis=1))

    return s_num,s_den

def tracking(image,target_position,window_size,num,den,cos_window,scalefactor):

    #window_size no change
    ori = get_ori(image,target_position,window_size,scalefactor)
    ori = pre_process_image(ori)
    feature_ensemble = get_feature(ori)

    for i in range(numlayers):

        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))
        x = ndimage.zoom(feature,(float(cos_window.shape[0])/feature.shape[0],float(cos_window.shape[1])/feature.shape[1],1),order=1)
        x = np.multiply(x,cos_window[:,:,None])
        xf = np.fft.fft2(x, axes=(0,1))
        response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(num[i],xf),axis=2),(den[i] + lam)))) * layerweights[i]

        if i == 0 :
            final_response = response
        else:
            final_response = np.add(final_response, response)

    center_h,center_w = np.unravel_index(final_response.argmax(),final_response.shape)

    h_delta,w_delta = [(center_h - final_response.shape[0]/2)*scalefactor*cell_size, (center_w - final_response.shape[1]/2)*scalefactor*cell_size]

    center = [target_position[0] + h_delta, target_position[1] + w_delta]

    return center

def scale_variation(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size):

    sw = get_scale_window(image,target_position,target_size,scale_factor*ScaleFactors,scale_window,model_size)
    swf = np.fft.fftn(sw,axes=[0])
    scale_response = np.real(np.fft.ifftn(np.sum(np.divide(np.multiply(scale_num,swf),(scale_den[:,None] + lam)),axis=1)))
    scale_index = np.argmax(scale_response)
    new_scale_factor = scale_factor * ScaleFactors[scale_index]

    return new_scale_factor

def update_position_filter(image,target_position,window_size,scale_factor,position_yf,position_cos_window,position_num,position_den,update_rate):

    ori = get_ori(image, target_position, window_size, scale_factor)
    ori = pre_process_image(ori)
    feature_ensemble = get_feature(ori)

    for i in range(numlayers):
        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))
        x = ndimage.zoom(feature,(float(position_cos_window.shape[0])/feature.shape[0],float(position_cos_window.shape[1])/feature.shape[1],1),order=1)
        x = np.multiply(x,position_cos_window[:,:,None])
        xf = np.fft.fft2(x,axes=(0,1))
        new_num = np.multiply(position_yf[:,:,None],np.conj(xf))
        new_den = np.real(np.sum(np.multiply(xf,np.conj(xf)),axis=2))

        position_num[i] = (1-update_rate)*position_num[i] + update_rate*new_num
        position_den[i] = (1-update_rate)*position_den[i] + update_rate*new_den

    return position_num,position_den

def update_scale_filter(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size,scale_ysf,update_rate):

    sw = get_scale_window(image,target_position, target_size, scale_factor*ScaleFactors, scale_window, model_size)
    swf = np.fft.fftn(sw, axes=[0])
    new_s_num = np.multiply(scale_ysf[:, None], np.conj(swf))
    new_s_den = np.real(np.sum(np.multiply(swf, np.conj(swf)), axis=1))

    scale_num = (1 - update_rate)*scale_num + update_rate*new_s_num
    scale_den = (1 - update_rate)*scale_den + update_rate*new_s_den

    return scale_num,scale_den

class tracker:
    def __init__(self, image, size, position):

        self.target_size = np.array([size[1], size[0]])

        self.pos = [position[1] + size[1] / 2, position[0] + size[0] / 2]
        self.sz = get_search_windows(self.target_size, image.shape[:2])

        # position prediction params
        self.lamda = 1e-4
        output_sigma_factor = 0.1
        self.cell_size = 4
        self.interp_factor = 0.01
        self.x_num = []
        self.x_den = []

        # scale estimation params
        self.current_scale_factor = 1.0
        nScales = 33
        scale_step = 1.02  # step of one scale level
        scale_sigma_factor = 1 / float(4)
        self.interp_factor_scale = 0.01
        scale_model_max_area = 32 * 16
        scale_model_factor = 1.0
        self.min_scale_factor = np.power(scale_step,
                                         np.ceil(np.log(5. / np.min(self.sz)) / np.log(scale_step)))

        self.max_scale_factor = np.power(scale_step,
                                         np.floor(np.log(np.min(np.divide(image.shape[:2],
                                                                          self.target_size)))
                                                  / np.log(scale_step)))

        if scale_model_factor * scale_model_factor * np.prod(self.target_size) > scale_model_max_area:
            scale_model_factor = np.sqrt(scale_model_max_area / np.prod(self.target_size))

        self.scale_model_sz = np.floor(self.target_size * scale_model_factor)

        # Gaussian shaped label for position perdiction
        l1_patch_num = np.floor(self.sz / self.cell_size)
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor / self.cell_size
        grid_y = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
        grid_x = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        self.yf = np.fft.fft2(y, axes=(0, 1))

        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # Gaussian shaped label for scale estimation
        ss = np.arange(nScales) - np.ceil(nScales / 2)
        scale_sigma = np.sqrt(nScales) * scale_sigma_factor
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.scaleFactors = np.power(scale_step, -ss)
        self.ysf = np.fft.fft(ys)
        if nScales % 2 == 0:
            self.scale_window = np.hanning(nScales + 1)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(nScales)

        # Extracting hierarchical convolutional features and training
        #get_filter(feature_ensemble,yf,cos_window)
        img = get_ori(image, self.pos, self.sz)
        self.pre_img = img

        img = pre_process_image(img)
        feature_ensemble = get_feature(img)

        self.x_num,self.x_den = get_filter(feature_ensemble,self.yf,self.cos_window)

        # Extracting the sample feature map for the scale filter and training
        #get_scale_filter(image,position,target_size,current_scale_factor,scaleFactors,scale_window,model_size,ysf)
        self.s_num,self.s_den = get_scale_filter(image,self.pos,self.target_size,self.current_scale_factor,self.scaleFactors,self.scale_window,self.scale_model_sz,self.ysf)



    def track(self, image):

            '''
            next_image = get_ori(image, self.pos, self.sz)

            w,h = get_optical_flow(self.pre_img,next_image)

            self.pre_img = next_image

            self.pos = [self.pos[0] + h, self.pos[1] + w]
            '''

            #tracking(image,target_position,window_size,num,den,cos_window,scalefactor)
            self.pos = tracking(image,self.pos,self.sz,self.x_num,self.x_den,self.cos_window,self.current_scale_factor)

            #scale_variation(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size)
            #self.current_scale_factor = scale_variation(image,self.pos,self.target_size,self.s_num,self.s_den,self.current_scale_factor,self.scaleFactors,self.scale_window,self.scale_model_sz)

            if self.current_scale_factor < self.min_scale_factor:
                self.current_scale_factor = self.min_scale_factor
            elif self.current_scale_factor > self.max_scale_factor:
                self.current_scale_factor = self.max_scale_factor

            # update
            #update_position_filter(image, target_position, window_size, scale_factor, position_yf, position_cos_window,
            #                      position_num, position_den, update_rate)
            self.x_num,self.x_den = update_position_filter(image,self.pos,self.sz,self.current_scale_factor,self.yf,self.cos_window,
                                    self.x_num,self.x_den,self.interp_factor)

            #update_scale_filter(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size,scale_ysf,update_rate)
            self.s_num,self.s_den = update_scale_filter(image,self.pos,self.target_size,self.s_num,self.s_den,self.current_scale_factor,self.scaleFactors,self.scale_window,self.scale_model_sz,self.ysf,self.interp_factor_scale)

            self.final_size = self.target_size * self.current_scale_factor

            target_box = (self.pos[1] - self.final_size[1] / 2,
                          self.pos[0] - self.final_size[0] / 2,
                          self.pos[1] + self.final_size[1] / 2,
                          self.pos[0] + self.final_size[0] / 2)

            return target_box


root = '/home/icv/PycharmProjects/TrackingData/otb100/MotorRolling'

image_path = os.path.join(root,'img')
if(os.path.exists(image_path) == False):
    image_path = os.path.join(root,'imgs')
image_file = os.listdir(image_path)
image_file.sort()

gt_file = os.path.join(root, 'groundtruth.txt')
if os.path.exists(gt_file) is False:
    gt_file = os.path.join(root,'groundtruth_rect.txt')

try:
    gt = np.loadtxt(gt_file)
except Exception, e:
    gt = np.loadtxt(gt_file, delimiter=",")

initial_frame = cv2.imread(os.path.join(image_path,image_file[0]))
im_size = (initial_frame.shape[0],initial_frame.shape[1])

initial_groundturth = gt[0]
position = [initial_groundturth[0],initial_groundturth[1]]
size = [initial_groundturth[2],initial_groundturth[3]]
size = np.array(size)

tracker = tracker(initial_frame,size,position)

image_file = image_file[1:]
gt = gt[1:]

#Tracking
for index in range(len(image_file)):

    print(image_file[index])
    tracking_image = cv2.imread(os.path.join(image_path,image_file[index]))

    region = tracker.track(tracking_image)

    show_image = Image.open(os.path.join(image_path,image_file[index]))
    draw = ImageDraw.Draw(show_image)

    draw.rectangle(region,outline='blue')

    groundturth = (gt[index][0], gt[index][1], gt[index][0] + gt[index][2], gt[index][1] + gt[index][3])
    draw.rectangle(groundturth, outline='green')

    show_image.save(os.path.join('result', image_file[index]))
