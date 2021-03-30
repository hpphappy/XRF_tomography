from mpi4py import MPI
import datetime
import numpy as np
from numpy.random import default_rng
import torch as tc
import os
import h5py
import sys
from misc_mpi_updating import print_flush_root
from data_generation_fns_mpi_updating_h5Parray import intersecting_length_fl_detectorlet_3d_mpi_write_h5

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

# def trace_beam_z(z_s, x_s, y_s, z_d, x_d, y_d, d_z_ls):
#     # For the case that the voxel and the detector have the same z coordinate, the connection of them doesn't have any intersection on any plane along z-direction.
#     if len(d_z_ls) == 0 or z_s == z_d:
#         Z = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
#     else:
#         A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[1, 0, 0, 0]])
#         A = A.repeat([len(d_z_ls), 1, 1])

#         b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_z_ls), 1, 1])
#         b2 = tc.tensor([[[d_z]] for d_z in d_z_ls])
#         b = tc.cat((b1, b2), dim=1)

#         Z, LU = tc.solve(b, A)
#         Z = np.array(Z[:,:-1].view(len(d_z_ls), 3))
# #         t = X[:,-1] 
    
#     return Z

# def trace_beam_x(z_s, x_s, y_s, z_d, x_d, y_d, d_x_ls):
#     # For the case that the voxel and the detector have the same x coordinate, the connection of them doesn't have any intersection on any plane along x-direction.
#     if len(d_x_ls) == 0:
#         X = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
#     else:    
#         A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[0, 1, 0, 0]])
#         A = A.repeat([len(d_x_ls), 1, 1])

#         b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_x_ls), 1, 1])
#         b2 = tc.tensor([[[d_x]] for d_x in d_x_ls])
#         b = tc.cat((b1, b2), dim=1)

#         X, LU = tc.solve(b, A)
#         X = np.array(X[:,:-1].view(len(d_x_ls), 3))
# #         t = Y[:,-1]
    
#     return X

# def trace_beam_y(z_s, x_s, y_s, z_d, x_d, y_d, d_y_ls):
#     # For the case that the voxel and the detector have the same y coordinate, the connection of them doesn't have any intersection on any plane along y-direction.
#     if len(d_y_ls) == 0 or y_s == y_d:
#         Y = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
#     else:
#         A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[0, 0, 1, 0]])
#         A = A.repeat([len(d_y_ls), 1, 1])

#         b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_y_ls), 1, 1])
#         b2 = tc.tensor([[[d_y]] for d_y in d_y_ls])
#         b = tc.cat((b1, b2), dim=1)

#         Y, LU = tc.solve(b, A)
#         Y = np.array(Y[:,:-1].view(len(d_y_ls), 3))
# #         t = Z[:,-1]
    
#     return Y

# def intersecting_length_fl_detectorlet_3d_mpi_write_h5(n_ranks, rank, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, P_folder, f_P):
#     """
#     Parameters
#     ----------
#     det_size_cm : float
#         The diameter of the circle to distribute the detector points
        
#     det_from_sample_cm : float
#         The distance between the detector plane and the sample boundary plane
    
#     det_ds_spacing_cm : float
#         The spacing between detector points
    
#     sample_size_n: int scalar
#         sample size in number of pixles on the side along the probe propagation axis
    
#     sample_size_cm: scalar
#         sample size in cm on the side along the probe propagation axis
        
#     sample_height_n : integer
#         The height of the sample along the rotational axis (in number of pixels)
        
#     P_save_path : string
#         The path that saves the tensor P

#     Returns
#     -------
#     n_det : integer
#         The number of the detector points within the circle with the diatmeter, det_size_cm.
    
#     P : torch tensor
#         a tensor with the dimension (n_det, 3, n_voxels *  diagnal_length_n)
#         n_voxels: the number of voxels of the sample.
#         diagnal_length_n: the number of voxels along the diagnol direction of the sample
        
#         P tensor contains the information of intersecting voxels of the emitted XRF rays (along the connection between each FL emitting source voxel and each detector point)
#         For each detector point (total: n_det), 3 rows of values representing the following values:
#             1st row, the index of the FL emitting soruce voxel. The index is the index of the flattened grid of the sample.
#             2nd row, the index of the intersecting voxels.
#             3rd row, the intersecting length in cm.
            
            
#             For example:
#                 [[0, 0, 0, 0, 0, 0, ..., 0, 1, 1, 1, 1, 0, ..., 0, 2, 2, 2, 0, ..., 0, ......, 0, ...,0]
#                                             |_________| \________|
#                                                       \          \The remain (diagnal_length_n - 4) spaces are then set to 0
#                                                       \4 intersecting voxels from the emitting source at index 1  
                 
#                  [5,10,15,20,25, 0, ..., 0, 6,11,16,21, 0, ..., 0, 7,12,17, 0, ..., 0, ......, 0, ...,0]
#                                             |_________| \________|
#                                                       \          \The remain (diagnal_length_n - 4) spaces are then set to 0
#                                                       \4 intersecting voxels at index 6, 11, 16, 21 from the emitting source at index 1  
                 
                 
#                  [0.1, 0.1, 0.1, 0.1, 0, 0, ..., 0, 0.2, 0.2, 0.2 ,0.2, 0, ..., 0, 0.3, 0.3, 0.3, 0, ..., 0, ......, 0, ...,0]]
#                                                     |_________________| \________|
#                                                       \                          \The remain (diagnal_length_n - 4) spaces are then set to 0
#                                                       \4 intersecting lengths corresponging to the intersecting voxels in the 2nd row of this tensor
                
#             The intersecting number of voxels from each source is not always the same. The maximal possible intersecting number of voxels
#             is the number of voxels along the diagnol direction of the sample.
#             Therefore, diagnal_length_n spaces are used to store the intersecting voxels for each emitting source.
#             In most cases, the number of intersecting voxels for each source voxel is less than diagnal_length_n, The remaining spaces are filled with zeros.
    
#     """
#     if rank == 0:
#         with open(os.path.join(P_folder, 'P_array_parameters.txt'), "w") as P_params:
#             P_params.write("det_size_cm = %f\n" %det_size_cm)
#             P_params.write("det_from_sample_cm = %f\n" %det_from_sample_cm)
#             P_params.write("det_ds_spacing_cm = %f\n" %det_ds_spacing_cm)
#             P_params.write("sample_size_n = %f\n" %sample_size_n)
#             P_params.write("sample_size_cm = %f\n" %sample_size_cm)
#             P_params.write("sample_height_n = %f\n" %sample_height_n)
            
#     layers_divisible_by_n_ranks = sample_height_n % n_ranks
#     if layers_divisible_by_n_ranks != 0:
#         print("Please set n_ranks such that sample_height_n is divisible by n_ranks")
         
#     P_save_path = os.path.join(P_folder, f_P)

#     if not os.path.exists(P_folder):
#         os.makedirs(P_folder)

#     ### Calculating voxel size in cm
#     voxel_size_cm = sample_size_cm/sample_size_n

#     ### Calculating the diameter of the XRF detector with 
#     det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

#     ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
#     det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)

#     # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
#     z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
#     voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


#     ### Define the location of the detectorlets, the detector is parallel to the yz-plane
#     ### The x-posision depends on the distance between the sample and the detecor
#     ## x index of the location of the XRF detector
#     det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5

#     ### y, z index of the location of the XRF detector
#     ## Define the center of the detector on yz-plane
#     det_center_yz = (int(sample_size_n)/2., int(sample_size_n)/2.)

#     ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
#     end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
#                                       int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

#     det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

#     end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
#                                       int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

#     det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
#     ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
#     y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

#     yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
#     y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


#     ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
#     x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)

#     ##
#     det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
#     n_det = len(det_pos_ls_flat)
    
#     if rank == 0:
#         print(f"numbder of detecting points: {n_det}")
#         sys.stdout.flush()
        
#     ## define sample edges: 
#     ## sample_x_edge is the edge that is closer to the XRF detector
#     ## sample_y_edge has two components representing the left and the right edge
#     sample_x_edge = np.array([sample_size_n])
#     sample_y_edge = np.array([0, sample_size_n]) 
#     sample_z_edge = np.array([0, sample_height_n]) 

#     dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
#     longest_int_length = 0
    
#     n_layers_each_rank = sample_height_n // n_ranks
#     voxel_pos_ls_flat_minibatch = voxel_pos_ls_flat[rank * n_layers_each_rank * sample_size_n**2 : (rank+1) * n_layers_each_rank * sample_size_n**2]
    
#     f = h5py.File(P_save_path +'.h5', 'w', driver='mpio', comm=comm)
#     P = f.create_dataset('P_array', (n_det, 3, dia_len_n * sample_height_n * sample_size_n**2), dtype='f4', data=np.zeros((n_det, 3, dia_len_n * sample_height_n * sample_size_n**2)))
    
    
#     j_offset = rank * n_layers_each_rank * sample_size_n**2
    
#     stdout_options = {'root':0, 'output_folder': './', 'save_stdout': False, 'print_terminal': True}
#     for i,  det_pos in enumerate(det_pos_ls_flat):
#         timestr = str(datetime.datetime.today())     
#         print_flush_root(rank, val=f"detecting point: {i}, time: {timestr}", output_file='', **stdout_options)
#         for j, v in enumerate(voxel_pos_ls_flat_minibatch): 

#             # Solving the intersection of the ray with the sample boundary along axis-0
#             bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge) # pick the 0th component just because the coordinate is doubly braced

#             # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
#             # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
#             bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
#             if len(bdy_int) != 0:
#                 bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
#             else:
#                 pass


#             bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
#             if len(bdz_int) != 0:
#                 bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
#             else:
#                 pass

#             # Pick the intersecting point that first hit the boundary plan. This point is with the least x value among the 3 intersections.
#             bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
#             bd_int = np.clip(np.abs((bd_int_ls[np.argmin(bd_int_ls[:,1])])), 0, sample_size_n)


#             # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
#             # find the x,y,z-value of the voxel boundary except the ones on the sample edge

#             z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
#                                                    np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

#             x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
#                                                    np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

#             y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
#                                                    np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))


#             z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
#             x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
#             y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)

#             # Collect all intersecting points and sort all intersections using the x coordinate
#             int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))     
#             int_ls = int_ls[np.argsort(int_ls[:,1])]

#             # calculate the intersecting length in the intersecting voxels
#             int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
#             # just in case that we count some intersections twice, delete the duplicates
#             idx_duplicate = np.array(np.where(int_length==0)).flatten()
#             int_ls = np.delete(int_ls, idx_duplicate, 0)
#             int_length = np.delete(int_length, idx_duplicate) 

#             # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
#             int_ls_shift = np.zeros((int_ls.shape))
#             int_ls_shift[1:] = int_ls[:-1]
#             int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
# #                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
#             int_idx_flat = int_idx[:,0] * (sample_height_n.item() * sample_size_n.item()) + int_idx[:,1] * sample_size_n.item() + int_idx[:,2]

#             if len(int_idx_flat) > longest_int_length:
#                 longest_int_length = len(int_idx_flat)
                
#             P[i, 0, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = j_offset+j
#             P[i, 1, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_idx_flat)
#             P[i, 2, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_length * voxel_size_cm.item())            

    
#     f_short = h5py.File(P_save_path +'_short.h5', 'w', driver='mpio', comm=comm)
#     P_short = f_short.create_dataset('P_short_array', (n_det, 3, longest_int_length * sample_height_n * sample_size_n**2), dtype='f4')
        
#     for j, v in enumerate(voxel_pos_ls_flat_minibatch):
#         P_short[:,:,(j_offset+j) * longest_int_length: (j_offset+j+1) * longest_int_length] = \
#         P[:,:, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + longest_int_length]
        
#     f.close()
#     f_short.close()
#     return longest_int_length, n_det, P


params = {
        "n_ranks" : n_ranks,
        "rank" : rank,
        "det_size_cm" : 0.9,
        "det_from_sample_cm" : 1.6,
        "det_ds_spacing_cm" : 0.4,
        "sample_size_n" : tc.tensor(64),
        "sample_size_cm" : tc.tensor(0.01),
        "sample_height_n" : tc.tensor(64),
        "P_folder" : 'data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/backup4',
        "f_P" : 'Intersecting_Length_64_64_64'
        }



longest_int_length, n_det, P = intersecting_length_fl_detectorlet_3d_mpi_write_h5(**params)
print(n_det)
sys.stdout.flush()