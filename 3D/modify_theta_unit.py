import python
import h5py
import os

# experiemtal parameters #
theta_st = 0
theta_end = 360
n_theta =  tc.tensor(800).to(dev)
sample_size_n = 256
sample_height_n = 256
sample_size_cm = 0.01
# this_aN_dic = {"Al": 13, "Si": 14, "Fe": 26, "Cu": 29}
this_aN_dic = {"Ca": 20, "Sc": 21}
probe_energy = np.array([20.0])
# det_size_cm = 0.9
# det_from_sample_cm = 1.6
# det_ds_spacing_cm = 0.4


# XRF and XRT data path #
data_path = './data/size_256'
f_XRF_data = 'simulation_XRF_data'
f_XRT_data = 'simulation_XRT_data'

with h5py.File(os.path.join(data_path, f_XRF_data +'.h5'), "r+") as s:
    XRF_data = s["exchange/data"][...].astype(np.float32)
    XRF_data = np.clip(XRF_data, 0, np.inf)
    s["exchange/data"][...] = XRF_data
    
    theta_ls_degree = -np.linspace(theta_st, theta_end, n_theta+1)[:-1]
    s["exchange/theta"][...] = theta_ls_degree
    print(s["exchange/theta"][...] )
