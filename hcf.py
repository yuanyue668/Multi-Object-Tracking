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

outputlayer = [17,26,35]
numlayers = len(outputlayer)
layerweights = [0.25,0.5,1]

lam = 1e-4
cell_size = 4
model = vgg.vgg(outputlayer=outputlayer)

model_dict = model.state_dict()

params = torch.load('vgg19-dcbb9e9d.pth')
load_dict = {k: v for k, v in params.items() if 'features' in k}
model_dict.update(load_dict)
model.load_state_dict(model_dict)
model.cuda()

def get_search_windows(size,im_size,scale_factor = None):

    if(size[0] / size[1] > 2):
        #For object with large height
        window_size = np.floor(np.multiply(size, [1+0.4,1+1.8]))

    elif np.prod(size)/np.prod(im_size) > 0.05:
        window_size = np.floor(size * (1 + 1))

    else:
        window_size = np.floor(size * (1 + 1.8))

    return window_size

def get_ori(image,postion,window_size):

    sz_ori = window_size

    y = np.floor(postion[0])-np.floor(window_size[0]/2) + np.arange(window_size[0], dtype=int)
    x = np.floor(postion[1])-np.floor(window_size[1]/2) + np.arange(window_size[1], dtype=int)

    x,y = x.astype(int),y.astype(int)

    #check bounds
    x[x < 0] = 0
    y[y < 0] = 0

    x[x >= image.shape[1]] = image.shape[1] - 1
    y[y >= image.shape[0]] = image.shape[0] - 1

    ori = image[np.ix_(y,x)]

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

    return num,den,cos_window.shape

def tracking(ori,num,den,label_size):

    ori = pre_process_image(ori)
    feature_ensemble = get_feature(ori)

    for i in range(numlayers):

        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))
        x = ndimage.zoom(feature,(float(label_size[0])/feature.shape[0],float(label_size[1])/feature.shape[1],1),order=1)

        cos_window = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
        x = np.multiply(x,cos_window[:,:,None])
        xf = np.fft.fft2(x, axes=(0,1))
        response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(num[i],xf),axis=2),(den[i] + lam)))) * layerweights[i]

        if i == 0 :
            final_response = response
        else:
            final_response = np.add(final_response, response)

    center_v,center_h = np.unravel_index(final_response.argmax(),final_response.shape)

    return center_v,center_h,final_response.shape

def update_filter(image, position,window_size,yf,cos_window,num,den,update_rate):

    ori = get_ori(image, position, window_size)
    ori = pre_process_image(ori)
    feature_ensemble = get_feature(ori)

    for i in range(numlayers):
        feature = feature_ensemble[i].data[0].cpu().numpy().transpose((1,2,0))
        x = ndimage.zoom(feature,(float(cos_window.shape[0])/feature.shape[0],float(cos_window.shape[1])/feature.shape[1],1),order=1)
        x = np.multiply(x,cos_window[:,:,None])
        xf = np.fft.fft2(x,axes=(0,1))
        num[i] = (1-update_rate)*num[i] + update_rate*np.multiply(yf[:,:,None],np.conj(xf))
        den[i] = (1-update_rate)*den[i] + update_rate*np.real(np.sum(np.multiply(xf,np.conj(xf)),axis=2))

    return num,den

def translate_img_center(v,h,map_size,position):

    v_delta, h_delta = [(v - map_size[0]/2)*cell_size, (h - map_size[1]/2)*cell_size]
    center = [position[0] + v_delta, position[1] + h_delta]

    return center

root = '/home/icv/PycharmProjects/TrackingData/otb100/Basketball'

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

initial_groundturth = gt[0]
position = [initial_groundturth[1]+initial_groundturth[3]/2,initial_groundturth[0]+initial_groundturth[2]/2]
size = [initial_groundturth[3],initial_groundturth[2]]
size = np.array(size)
window_size = get_search_windows(size,initial_frame.shape[:2])

#Gaussian shaped label for position perdiction
l1_patch_num = np.floor(window_size / cell_size)
output_sigma = np.sqrt(np.prod(size)) * 0.1 / cell_size

grid_y = np.arange(np.floor(l1_patch_num[0])) - np.floor(l1_patch_num[0] / 2)
grid_x = np.arange(np.floor(l1_patch_num[1])) - np.floor(l1_patch_num[1] / 2)
'''
Specific comment: If we change the x and y, it also can tracking well when have no challges,
But when occlusion happend,it will shift that change to tracking the screen
'''
rs, cs = np.meshgrid(grid_x, grid_y)
y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
yf = np.fft.fft2(y, axes=(0, 1))
cos_window = np.outer(np.hanning(yf.shape[0]), np.hanning(yf.shape[1]))

#Filter Inital
ori = get_ori(initial_frame,position,window_size)
ori = pre_process_image(ori)
feature_ensemble = get_feature(ori)
num,den,filter_shape = get_filter(feature_ensemble,yf,cos_window)

image_file = image_file[1:]
gt = gt[1:]

#Tracking
for index in range(len(image_file)):

    print(image_file[index])
    tracking_image = cv2.imread(os.path.join(image_path,image_file[index]))

    tracking_ori = get_ori(tracking_image, position, window_size)

    center_v, center_h, response_shape = tracking(tracking_ori, num, den, filter_shape)
    position = translate_img_center(center_v, center_h, response_shape, position)
    num, den = update_filter(tracking_image,position,window_size,yf,cos_window, num, den, 0.01)

    show_image = Image.open(os.path.join(image_path,image_file[index]))
    draw = ImageDraw.Draw(show_image)
    traget = (position[1] - size[1]/2, position[0]-size[0]/2, position[1]+size[1]/2, position[0]+size[0]/2)
    draw.rectangle(traget,outline="red")
    show_image.save(os.path.join('result', image_file[index]))
    '''
    figure = plt.figure()
    plt.imshow(show_image)
    plt.show()
    '''
