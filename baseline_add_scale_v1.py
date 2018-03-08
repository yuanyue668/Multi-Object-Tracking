import os
import sys
import cv2
import vgg
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


def get_search_windows(size,im_size):

    if(size[1] / size[0] > 2):
        #For object with large height
        window_size = np.floor(np.multiply(size, [1+1.8,1+0.4]))

    elif np.prod(size)/np.prod(im_size) > 0.05:
        window_size = np.floor(size * (1 + 1))

    else:
        window_size = np.floor(size * (1 + 1.8))

    return window_size


def get_ori(image,position,wsz,scale_factor = None):

    sz_ori = wsz

    patch_wsz = wsz
    if scale_factor != None:
        patch_wsz = np.floor(patch_wsz*scale_factor)

    x = np.floor(position[0])-np.floor(patch_wsz[0]/2) + np.arange(patch_wsz[0], dtype=int)
    y = np.floor(position[1])-np.floor(patch_wsz[1]/2) + np.arange(patch_wsz[1], dtype=int)

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

def get_scale_window(image,position,tsz,sfs,scale_window,scale_model_sz):

    out = []
    for i in range(len(sfs)):
        patch_sz = np.floor(tsz * sfs[i])
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

    w_delta, h_delta = [(center_w - final_response.shape[1]/2)*scalefactor*cell_size, (center_h - final_response.shape[0]/2)*scalefactor*cell_size]

    center = [target_position[0] + w_delta, target_position[1] + h_delta]

    return center

def scale_variation(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size):

    sw = get_scale_window(image,target_position,target_size,scale_factor*ScaleFactors,scale_window,model_size)
    swf = np.fft.fftn(sw,axes=[0])
    scale_response = np.real(np.fft.ifftn(np.sum(np.divide(np.multiply(scale_num,swf),(scale_den[:,None] + lam)),axis=1)))
    scale_index = np.argmax(scale_response)
    new_scale_factor = scale_factor * ScaleFactors[scale_index]

    if new_scale_factor < min_scale_factor:
        new_scale_factor = min_scale_factor
    elif new_scale_factor > max_scale_factor:
        new_scale_factor = max_scale_factor

    new_target_size = target_size * new_scale_factor

    return new_target_size,new_scale_factor

def update_position_filter(image,target_position,window_size,scale_factor,position_yf,position_cos_window,position_num,position_den,update_rate):

    ori = get_ori(image, target_position, window_size, scale_factor)
    ori = pre_process_image(ori)
    feature_ensemble = get_feature(ori)

    for i in range(numlayers):
        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))
        x = ndimage.zoom(feature,(float(position_cos_window.shape[0])/feature.shape[0],float(position_cos_window.shape[1])/feature.shape[1],1),order=1)
        x = np.multiply(x,position_cos_window[:,:,None])
        xf = np.fft.fft2(x,axes=(0,1))
        position_num[i] = (1-update_rate)*position_num[i] + update_rate*np.multiply(position_yf[:,:,None],np.conj(xf))
        position_den[i] = (1-update_rate)*position_den[i] + update_rate*np.real(np.sum(np.multiply(xf,np.conj(xf)),axis=2))

    return position_num,position_den

def update_scale_filter(image,target_position,target_size,scale_num,scale_den,scale_factor,ScaleFactors,scale_window,model_size,scale_ysf,update_rate):

    sw = get_scale_window(image,target_position, target_size, scale_factor*ScaleFactors, scale_window, model_size)
    swf = np.fft.fftn(sw, axes=[0])
    new_s_num = np.multiply(scale_ysf[:, None], np.conj(swf))
    new_s_den = np.real(np.sum(np.multiply(swf, np.conj(swf)), axis=1))

    scale_num = (1 - update_rate)*scale_num + update_rate*new_s_num
    scale_den = (1 - update_rate)*scale_den + update_rate*new_s_den

    return scale_num,scale_den


outputlayer = [17,26,35]
numlayers = len(outputlayer)
layerweights = [0.25,0.5,1]
assert (numlayers == len(layerweights))

#Position Gaussian shaped label param
lam = 1e-4
cell_size = 4

model = vgg.vgg(outputlayer=outputlayer)
model_dict = model.state_dict()
params = torch.load('vgg19-dcbb9e9d.pth')
load_dict = {k: v for k, v in params.items() if 'features' in k}
model_dict.update(load_dict)
model.load_state_dict(model_dict)
model.cuda()

root = '/home/icv/PycharmProjects/TrackingData/otb100/Dog1'

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
position = [initial_groundturth[0]+initial_groundturth[2]/2,initial_groundturth[1]+initial_groundturth[3]/2]
size = [initial_groundturth[2],initial_groundturth[3]]
size = np.array(size)
window_size = get_search_windows(size,initial_frame.shape[:2])

#Gaussian shaped label for position perdiction
l1_patch_num = np.floor(window_size / cell_size)
output_sigma = np.sqrt(np.prod(size)) * 0.1 / cell_size
grid_x = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
grid_y = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
rs, cs = np.meshgrid(grid_x, grid_y)
y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
yf = np.fft.fft2(y, axes=(0, 1))
cos_window = np.outer(np.hanning(yf.shape[0]), np.hanning(yf.shape[1]))


#Scale Gaussian shaped label param
nScales = 33
scale_step = 1.02
current_scale_factor = 1.0
scale_sigma_factor = 1 / float(4)
scale_model_max_area = 32 * 16
scale_model_factor = 1.0

#Gaussian shaped lable for scale perdiction
min_scale_factor = np.power(scale_step,np.ceil(np.log(5./np.min((window_size[1],window_size[0])))/ np.log(scale_step)))
max_scale_factor = np.power(scale_step,np.floor(np.log(np.min(np.divide(initial_frame.shape[:2],(size[1],size[0]))))/np.log(scale_step)))

if scale_model_factor*scale_model_factor*np.prod(size) > scale_model_max_area:
    scale_model_factor = np.sqrt(scale_model_max_area / np.prod(size))
scale_model_sz = np.floor(size * scale_model_factor)

ss = np.arange(nScales) - np.ceil(nScales / 2)
scale_sigma = np.sqrt(nScales) * scale_sigma_factor
scaleFactors = np.power(scale_step, -ss)
ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
ysf = np.fft.fft(ys)

if nScales % 2 == 0:
    scale_window = np.hanning(nScales + 1)
    scale_window = scale_window[1:]
else:
    scale_window = np.hanning(nScales)

#Filter Inital
ori = get_ori(initial_frame,position,window_size)
ori = pre_process_image(ori)
feature_ensemble = get_feature(ori)
num,den = get_filter(feature_ensemble,yf,cos_window)

#Scale Filter Inital
s_num,s_den = get_scale_filter(initial_frame,position,size,current_scale_factor,scaleFactors,scale_window,scale_model_sz,ysf)

image_file = image_file[1:]
gt = gt[1:]

#Tracking
for index in range(len(image_file)):

    print(str(image_file[index])+'...Current factor is '+str(current_scale_factor))
    tracking_image = cv2.imread(os.path.join(image_path,image_file[index]))

    position = tracking(tracking_image,position,window_size,num,den,cos_window,current_scale_factor)
    target_size,current_scale_factor = scale_variation(tracking_image,position,size,s_num,s_den,current_scale_factor,scaleFactors,scale_window,scale_model_sz)

    num, den = update_position_filter(tracking_image,position,window_size,current_scale_factor,yf,cos_window,num,den,0.01)
    s_num, s_den = update_scale_filter(tracking_image,position,size,s_num,s_den,current_scale_factor,scaleFactors,scale_window,scale_model_sz,ysf,0.01)

    show_image = Image.open(os.path.join(image_path,image_file[index]))
    draw = ImageDraw.Draw(show_image)
    target = (position[0]-target_size[0]/2,position[1]-target_size[1]/2,position[0]+target_size[0]/2,position[1]+target_size[1]/2)
    #noScale_target = (position[0]-size[0]/2,position[1]-size[1]/2,position[0]+size[0]/2,position[1]+size[1]/2)
    groundturth = (gt[index][0],gt[index][1],gt[index][0]+gt[index][2],gt[index][1]+gt[index][3])
    draw.rectangle(target,outline="red")
    #draw.rectangle(noScale_target,outline='blue')
    draw.rectangle(groundturth, outline='green')
    show_image.save(os.path.join('result', image_file[index]))

