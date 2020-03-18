import numpy as np
import torch
import cv2
from pysot.core.config import cfg
'''numpy version'''
def get_subwindow_numpy(im, pos, model_sz, original_sz, avg_chans):
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
    return im_patch
'''pytorch version (fully differentiable)'''
def get_subwindow(im, pos, model_sz, original_sz, avg_chans):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = [im.size(2),im.size(3)]
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

    _, k, r, c  = im.size()
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (1, k, r + top_pad + bottom_pad, c + left_pad + right_pad)
        te_im = torch.zeros(size)
        te_im[:, :, top_pad:top_pad + r, left_pad:left_pad + c] = im
        if top_pad:
            te_im[:, :, 0:top_pad, left_pad:left_pad + c] = avg_chans
        if bottom_pad:
            te_im[:, :, r + top_pad:, left_pad:left_pad + c] = avg_chans
        if left_pad:
            te_im[:, :, :, 0:left_pad] = avg_chans
        if right_pad:
            te_im[:, :, :, c + left_pad:] = avg_chans
        im_patch = te_im[:, :, int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1)]
    else:
        im_patch = im[:, :, int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1)]

    if not np.array_equal(model_sz, original_sz):
        im_patch = torch.nn.functional.interpolate(im_patch, size=(model_sz,model_sz),mode='bilinear')
    # im_patch = im_patch.transpose(2, 0, 1)  # shape: (H,W,3)-->(3,H,W)
    # im_patch = im_patch[np.newaxis, :, :, :]  # shape: (1,3,H,W)
    # im_patch = im_patch.astype(np.float32)
    # im_patch = torch.from_numpy(im_patch)
    if cfg.CUDA:
        im_patch = im_patch.cuda()
    return im_patch
