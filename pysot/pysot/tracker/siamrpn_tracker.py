# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from data_utils import tensor2img
import cv2
import os
''' Adversarial Attack tools :)'''
from attack_utils import adv_attack_template, adv_attack_search, \
    add_gauss_noise, add_pulse_noise, adv_attack_template_S

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        # print(score.size()) #(1,2x5,25,25)
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_z_crop(self, img, bbox):
        """
                args:
                    img(np.ndarray): BGR image
                    bbox: (x, y, w, h) bbox
                """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z_crop

    def get_x_crop(self,img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        return x_crop, scale_z

    def x_crop_2_res(self, img, x_crop, scale_z):
        outputs = self.model.track(x_crop)
        '''post-process'''
        score = self._convert_score(outputs['cls'])  # (25x25x5,)
        # print(score.shape)
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        self.score = score # newly added
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def save_img(self, tensor_clean, tensor_adv, save_path, frame_id):
        ## clean x_crop
        img_clean = tensor2img(tensor_clean)
        cv2.imwrite(os.path.join(save_path, '%04d_clean.jpg' % frame_id), img_clean)
        ## adv x_crop
        img_adv = tensor2img(tensor_adv)
        cv2.imwrite(os.path.join(save_path, '%04d_adv.jpg' % frame_id), img_adv)
        ## diff
        tensor_diff = (tensor_adv - tensor_clean) * 10
        # print(torch.mean(torch.abs(tensor_diff)))
        tensor_diff += 127.0
        img_diff = tensor2img(tensor_diff)
        cv2.imwrite(os.path.join(save_path, '%04d_diff.jpg' % frame_id), img_diff)

    def init(self, img, bbox):
        z_crop = self.get_z_crop(img,bbox)
        self.model.template(z_crop)

    def init_adv(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.model.template(z_crop_adv)
        '''save'''
        if save_path != None and name != None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name+'_clean.jpg'),z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def init_adv_S(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template_S(z_crop, GAN)
        self.model.template(z_crop_adv)
        '''save'''
        if save_path != None and name != None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name+'_clean.jpg'),z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def track(self, img):
        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        return output_dict

    def track_adv(self, img, GAN, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        '''Adversarial Attack'''
        x_crop_adv = adv_attack_search(x_crop, GAN)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path!=None and frame_id!=None:
            '''save'''
            self.save_img(x_crop,x_crop_adv,save_path,frame_id)
        return output_dict

    def track_gauss(self, img, sigma, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        '''Gaussian Attack'''
        x_crop_adv = add_gauss_noise(x_crop,sigma)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path!=None and frame_id!=None:
            '''save'''
            self.save_img(x_crop,x_crop_adv,save_path,frame_id)
        return output_dict

    def track_impulse(self, img, prob, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        '''impulse Attack'''
        x_crop_adv = add_pulse_noise(x_crop, prob)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path!=None and frame_id!=None:
            '''save'''
            self.save_img(x_crop,x_crop_adv,save_path,frame_id)
        return output_dict

    def track_heatmap(self, img):
        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        '''process score map'''
        score_map = np.max(self.score.reshape(5, 25, 25), axis=0)
        return output_dict, score_map

    '''supplementary material'''
    def track_supp(self, img, GAN, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        '''save clean region and heatmap'''
        x_crop_img = tensor2img(x_crop)
        cv2.imwrite(os.path.join(save_path, 'ori_search_%d.jpg' % frame_id), x_crop_img)
        '''original heatmap'''
        outputs_clean = self.model.track(x_crop)
        score = self._convert_score(outputs_clean['cls']) #(25x25x5,)
        heatmap_clean = 255.0*np.max(score.reshape(5, 25, 25), axis=0)#[0,1]
        heatmap_clean = cv2.resize(heatmap_clean,(255,255),interpolation=cv2.INTER_CUBIC)
        heatmap_clean = cv2.applyColorMap(heatmap_clean.clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_clean_%d.jpg' % frame_id), heatmap_clean)
        '''Adversarial Attack'''
        x_crop_adv = adv_attack_search(x_crop, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        '''save adv region and heatmap'''
        x_crop_img_adv = tensor2img(x_crop_adv)
        cv2.imwrite(os.path.join(save_path, 'adv_search_%d.jpg' % frame_id), x_crop_img_adv)
        score_adv = self.score
        heatmap_adv = 255.0 * np.max(score_adv.reshape(5, 25, 25), axis=0)  # [0,1]
        heatmap_adv = cv2.resize(heatmap_adv, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_adv = cv2.applyColorMap(heatmap_adv.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_adv_%d.jpg' % frame_id), heatmap_adv)
        return output_dict
