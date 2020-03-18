import cv2
import torch
import numpy as np
from attack_utils import add_pulse_noise, add_gauss_noise

if __name__ == '__main__':

    base = torch.ones((255,255,3)) * 127
    base = base.cuda()
    '''gauss noise'''
    sigma = 0.1
    gauss_vis = add_gauss_noise(base,sigma)
    '''impulse noise'''
    prob = 0.2
    impulse_vis = add_pulse_noise(base,prob)
    def tensor2img(tensor):
        return np.array(tensor.cpu()).clip(0,255).astype(np.uint8)
    gauss_img = tensor2img(gauss_vis)
    impulse_img = tensor2img(impulse_vis)
    cv2.imwrite('gauss_noise.jpg',gauss_img)
    cv2.imwrite('impulse_noise.jpg',impulse_img)



