import open3d as o3d
from tof2image import Bin_2_Array
import numpy as np

bin_input_fld = "D:/media/ssd/ssd_data/Experimentos/bin_exp1"
video_buffer=Bin_2_Array(bin_input_fld).reshaped_grouped()


reference=video_buffer[5,:,:,:]
for frame in range(video_buffer.shape[0]):
        
    
    pcd_round=np.round(pcloud_arr,4)*1000
    from IPython import embed; embed()
    relative_amp = 
