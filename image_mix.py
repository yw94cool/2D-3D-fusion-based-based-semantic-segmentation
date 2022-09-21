import glob
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np

top_file_path = 'D:/DeepSEG/test1/top/*.jpg'
top_file_list = glob.glob(top_file_path)


for f in top_file_list:
    f_name = f.split('\\')[-1]
    save_name = f_name.split('.')[0]
    print(save_name)

    f_subname = f_name.split('_')
    f_len = len(f_subname)


    if len(f_subname) == 5:
        dsm_name = 'dsm_' + f_subname[1] + '_' + f_subname[2] + '_' + f_subname[3] + '_' + f_subname[4]
    elif len(f_subname) == 6:
        dsm_name = 'dsm_' + f_subname[1] + '_' + f_subname[2] + '_' + f_subname[3] + '_' + f_subname[4] + '_' + f_subname[5]
    elif len(f_subname) == 7:
        dsm_name = 'dsm_' + f_subname[1] + '_' + f_subname[2] + '_' + f_subname[3] + '_' + f_subname[4] + '_' + f_subname[5] + '_' + f_subname[6]


    dsm_file_path = 'D:/DeepSEG/test1/dsm/' + dsm_name

    if not os.path.exists(dsm_file_path):
        raise IOError("DSM path:"+dsm_file_path+" not found")
             
    img_dsm = misc.imread(dsm_file_path)
    print (img_dsm.shape)
    img_rgb = misc.imread(f)
    print (img_rgb.shape)
    img_mix = np.dstack((img_rgb, img_dsm))
    save_path = 'D:/DeepSEG/test1/' + save_name
    np.save(save_path, img_mix)