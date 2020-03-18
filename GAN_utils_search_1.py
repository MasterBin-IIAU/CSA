from options.test_options0 import TestOptions
from models import create_model
import os
from common_path import project_path_
opt = TestOptions().parse()

# modify some config
'''Attack Search Regions'''
# opt.model = 'G_search_resize_512' # only cooling
opt.model = 'G_search_resize_512_regress' # cooling + shrinking
# opt.model = 'G_search_L2_500'
# opt.model = 'G_search_L2_500_regress'
opt.netG = 'unet_256'

# create and initialize model
'''create perturbation generator'''
GAN = create_model(opt)  # create a model given opt.model and other options
#GAN.load_path = os.path.join(project_path_,'checkpoints/%s/1_net_G.pth'%opt.model)
GAN.load_path = os.path.join('/home/masterbin-iiau/Desktop/AdvTrack-checkpoints','checkpoints/%s/1_net_G.pth'%opt.model)
GAN.setup(opt)  # # regular setup: load and print networks; create schedulers
GAN.eval()