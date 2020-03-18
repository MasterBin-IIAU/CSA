import torch
from data_utils import normalize
import numpy as np
from pysot.core.config import cfg
import cv2
'''This module is used to implement AA to template or search region'''

# def adv_attack_template(img_tensor,GAN):
#     '''adversarial attack to template'''
#     '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
#     '''step1: Normalization'''
#     img_tensor = normalize(img_tensor)
#     '''step2: pass to G'''
#     with torch.no_grad():
#         GAN.template_clean1 = img_tensor
#         GAN.forward()
#     img_adv = GAN.template_adv255
#     return img_adv
def adv_attack_template(img_tensor, GAN):
    '''adversarial attack to template'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        GAN.template_clean1 = img_tensor
        GAN.forward()
    img_adv = GAN.template_adv255
    return img_adv
def adv_attack_template_S(img_tensor, GAN, target_sz=(127,127)):
    '''adversarial attack to template'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        img_adv = GAN.transform(img_tensor,target_sz)
        return img_adv
def adv_attack_search(img_tensor,GAN,search_sz=(255,255)):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        GAN.search_clean1 = img_tensor
        GAN.num_search = img_tensor.size(0)
        GAN.forward(search_sz)
    img_adv = GAN.search_adv255
    return img_adv

def adv_attack_search_new(img_tensor,GAN,search_sz=(255,255)):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        GAN.tensor_clean1 = img_tensor
        GAN.num_search = img_tensor.size(0)
        GAN.forward(search_sz)
    img_adv = GAN.tensor_adv255
    return img_adv
def get_subwindow(im, pos, model_sz, original_sz, avg_chans, type='tensor'):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                         int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                      int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    if type == 'tensor':
        '''numpy array ---> torch tensor'''
        # cv2.imwrite('./hahaha.jpg',im_patch)
        im_patch = im_patch.transpose(2, 0, 1)#shape: (H,W,3)-->(3,H,W)
        im_patch = im_patch[np.newaxis, :, :, :]#shape: (1,3,H,W)
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
    return im_patch
def multi_cropx(img, center_pos, size, channel_average, type='tensor'):
    '''crop search region'''
    '''pos:(N,2); size:(N,2)'''
    w_z = size[:,0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size,axis=1)
    h_z = size[:,1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size,axis=1)
    s_z = np.sqrt(w_z * h_z)
    # scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    s_x = np.round(s_x)
    '''2019.10.10 remove small bbox'''
    mask = s_x > 5
    s_x = s_x[mask]
    center_pos = center_pos[mask]
    num_box = center_pos.shape[0]
    scale_x = 512 / s_x
    if type == 'tensor':
        x_crop = torch.zeros((num_box,3,512,512)).cuda()
    elif type == 'array':
        x_crop = np.zeros((num_box,512,512,3))

    for i in range(s_x.shape[0]):
        x_crop[i] = get_subwindow(img, center_pos[i],
                                512,
                                s_x[i], channel_average,type)
    return x_crop, scale_x
def add_gauss_noise(input, sigma):
    gauss_noise = torch.randn(input.size()) * (sigma * 255)
    gauss_noise = gauss_noise.cuda()
    output = input + gauss_noise
    output = output.clamp(0, 255)
    return output
def generate_impulse_mask(im_sz,prob):
    rdn = torch.rand(im_sz)
    mask0 = rdn < (prob / 2)
    mask1 = rdn > (1 -prob / 2)
    return mask0.cuda(), mask1.cuda()
def add_pulse_noise(input,prob):
    mask0, mask1 = generate_impulse_mask(input.size(),prob)
    output = input.clone()
    output[mask0] = 0
    output[mask1] = 255
    return output
