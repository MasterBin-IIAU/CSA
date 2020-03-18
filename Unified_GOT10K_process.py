import os
import glob
import numpy as np
import cv2
from pysot.core.config import cfg
from siamrpnpp_utils import get_subwindow_numpy

'''sampling interval'''
interval = 10 # as the paper uses

def crop_z(img, bbox, channel_average):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                bbox[1] + (bbox[3] - 1) / 2])
    size = np.array([bbox[2], bbox[3]])

    # calculate z crop size
    w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    s_z = round(np.sqrt(w_z * h_z))

    # get crop with s_z
    z_crop = get_subwindow_numpy(img, center_pos,
                                cfg.TRACK.EXEMPLAR_SIZE,
                                s_z, channel_average)
    return z_crop
def crop_x(img, bbox, channel_average):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                bbox[1] + (bbox[3] - 1) / 2])
    size = np.array([bbox[2], bbox[3]])
    w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    s_z = np.sqrt(w_z * h_z)
    '''s_x/s_z = cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE'''
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

    # get crop with s_x
    x_crop = get_subwindow_numpy(img, center_pos,
                                 cfg.TRACK.INSTANCE_SIZE,
                                 round(s_x), channel_average)
    return x_crop

if __name__ == '__main__':
    '''change following two paths to yours!'''
    # SSD is preferred for higher speed :)
    got10k_path = '/media/masterbin-iiau/WIN_SSD/GOT10K'
    save_path = '/media/masterbin-iiau/WIN_SSD/GOT10K_reproduce'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_path = os.path.join(got10k_path,'train')
    video_list = sorted(os.listdir(train_path))
    video_list.remove('list.txt')
    num_video = len(video_list)
    init_arr = np.zeros((num_video,4))
    for i ,video in enumerate(video_list):
        save_dir = os.path.join(save_path,video)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        frames_list = sorted(glob.glob(os.path.join(train_path,video,'*.jpg')))
        gt_file = os.path.join(train_path,video,'groundtruth.txt')
        gt_arr = np.loadtxt(gt_file,dtype=np.float64,delimiter=',')
        '''merge init gt into one file'''
        init_arr[i] = gt_arr[0].copy()
        init_frame = cv2.imread(frames_list[0])
        channel_average = np.mean(init_frame,axis=(0,1))
        '''crop & save initial template region'''
        z_crop = crop_z(init_frame, gt_arr[0], channel_average)
        dest_path = frames_list[0].replace(train_path, save_path)
        cv2.imwrite(dest_path, z_crop)
        '''crop search region every interval frames'''
        num_frames = len(frames_list)
        index_list = list(range(num_frames))[1:num_frames:interval]
        for index in index_list:
            frame_path = frames_list[index]
            cur_frame = cv2.imread(frame_path)
            x_crop = crop_x(cur_frame, gt_arr[index-1],channel_average)
            dest_path = frames_list[index].replace(train_path,save_path)
            cv2.imwrite(dest_path,x_crop)
        print('%d/%d completed.'%(i+1,num_video))
    save_file = os.path.join(save_path,'init_gt.txt')
    np.savetxt(save_file,init_arr,fmt='%.4f',delimiter=',')
