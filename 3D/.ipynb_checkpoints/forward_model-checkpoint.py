import numpy as np
import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import xraylib as xlib
import xraylib_np as xlib_np
import torch.nn as nn
from data_generation_fns import rotate, MakeFLlinesDictionary, intersecting_length_fl_detectorlet_3d


fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE])

fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE])

fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

class PPM(nn.Module):
    
    fl_line_groups=np.array(["K", "L", "M"]) 
    fl_K=fl_K
    fl_L=fl_L
    fl_M=fl_M
    group_lines=True
    
    def __init__(self, dev, ini_kind, init_const, grid_concentration, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                 this_aN_dic, probe_energy, probe_cts, fl_line_groups, fl_K, fl_L, fl_M, group_lines, 
                 theta_st, theta_end, n_theta, this_theta_idx,
                 det_ds_spacing_cm, det_size_cm, det_from_sample_cm, P_save_path):
        """
        Initialize the attributes of PPM. 
        """
        super(PPM, self).__init__() # inherit the __init__ from nn.Module.
        self.dev = dev
        self.ini_kind = ini_kind
        self.init_const = init_const
        self.grid_concentration = grid_concentration
        self.theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1]
        self.this_theta_idx = this_theta_idx
        self.n_element = n_element
        self.sample_height_n = sample_height_n
        self.minibatch_size = minibatch_size
        self.sample_size_n = sample_size_n
        self.p = p  # indicate which minibatch to calculate the gradient  
        self.xp = self.init_xp() # initialize the values of the minibatch
        
        self.probe_energy = probe_energy  
        self.this_aN_dic = this_aN_dic
        self.n_element = tc.as_tensor(len(self.this_aN_dic)).to(self.dev)
        self.element_ls = np.array(list(this_aN_dic.keys()))
        self.aN_ls = np.array(list(this_aN_dic.values()))
        
        self.probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.probe_energy).flatten()).to(self.dev)
#         self.probe_attCS_dic = dict(zip(self.element_ls, self.probe_attCS_ls))
        
        self.sample_size_cm = sample_size_cm
        self.fl_line_groups = fl_line_groups
        self.fl_K = fl_K
        self.fl_L = fl_L
        self.fl_M = fl_M
        self.group_lines = group_lines
        self.fl_all_lines_dic = self.init_fl_all_lines_dic()
        self.n_lines = tc.as_tensor(self.fl_all_lines_dic["n_lines"]).to(self.dev)
        self.FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.fl_all_lines_dic["fl_energy"])).float().to(self.dev)
        self.detected_fl_unit_concentration = tc.as_tensor(self.fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(self.dev)
        self.n_line_group_each_element = tc.IntTensor(self.fl_all_lines_dic["n_line_group_each_element"]).to(self.dev)
        
        self.dia_len_n = int((self.sample_height_n**2 + self.sample_size_n**2 + self.sample_size_n**2)**0.5)
        self.n_voxel_batch = self.minibatch_size * self.sample_size_n
        self.n_voxel = self.sample_height_n * self.sample_size_n**2   
        
        self.det_ds_spacing_cm = det_ds_spacing_cm
        self.det_size_cm = det_size_cm
        self.det_from_sample_cm = det_from_sample_cm
        self.P_save_path = P_save_path
        
        self.n_det, self.P_batch = self.init_intersecting_length_fl_detectorlet()  
        
        self.probe_cts = probe_cts
        self.probe_before_attenuation_flat = self.init_probe()

        
    def init_xp(self):
        """
        Initialize self.xp with the tensor of the true model(noise may already be added) (n_element, minibatch_size, n_y)
        """
        if self.ini_kind == "rand" or self.ini_kind == "randn":
            theta = self.theta_ls[self.this_theta_idx]
            concentration_map_rot = rotate(self.grid_concentration, theta, self.dev).view(self.n_element, self.sample_height_n * self.sample_size_n, self.sample_size_n)
            return nn.Parameter(concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :])
        
        if self.ini_kind == "const":        
            return nn.Parameter(tc.zeros(self.n_element, self.minibatch_size, self.sample_size_n) + self.init_const)

    def init_fl_all_lines_dic(self):
        """
        Initialize self.fl_all_lines_dic
        """
        fl_all_lines_dic = MakeFLlinesDictionary(self.this_aN_dic, self.probe_energy,
                      self.sample_size_n.cpu().numpy(), self.sample_size_cm.cpu().numpy(),
                      self.fl_line_groups, self.fl_K, self.fl_L, self.fl_M,
                      self.group_lines)
        return fl_all_lines_dic
    
    def init_intersecting_length_fl_detectorlet(self):
        """
        Initialize self.intersecting_length_fl_detectorlet
        """
        n_det, P = intersecting_length_fl_detectorlet_3d(self.det_size_cm, self.det_from_sample_cm, self.det_ds_spacing_cm,
                                                  self.sample_size_n.cpu().numpy(), self.sample_size_cm.cpu().numpy(),
                                                  self.sample_height_n.cpu().numpy(), self.P_save_path)
   
        P = P.float().to(self.dev)      
        P = P.view(n_det, 3, self.dia_len_n * self.sample_height_n * self.sample_size_n, self.sample_size_n)
        P_batch = P[:, :, self.dia_len_n * self.minibatch_size * self.p : self.dia_len_n * self.minibatch_size * (self.p+1), :].detach().clone()
        P_batch = P_batch.view(n_det, 3, self.dia_len_n * self.minibatch_size * self.sample_size_n)
        del P

        return n_det, P_batch

        return n_det, P_batch
    
    def init_probe(self):
        return self.probe_cts * tc.ones((self.minibatch_size * self.sample_size_n), device=self.dev)
    
    def forward(self, grid_concentration, this_theta_idx): 
        """
        Forward propagation.
        """      
        
        ### 1: Calculate the map of attenuation and transmission ###          
        theta = self.theta_ls[this_theta_idx]

        # Rotate the sample
        concentration_map_rot = rotate(grid_concentration, theta, self.dev).view(self.n_element, self.sample_height_n * self.sample_size_n, self.sample_size_n)
        
        # Set part of the sample to be the updating target
        concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :] = self.xp
        
        # Create a tensor to store the current batch of the sample
        concentration_map_rot_batch = concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :]
        
        ## Calculate the attenuation of the probe
        # Calculate the expoenent of attenuation of each voxel in the batch. (The atteuation before the probe enters each voxel.)
        att_exponent_acc_map = tc.zeros((self.minibatch_size, self.sample_size_n+1), device=self.dev)
        
        fl_map_tot_flat_theta = tc.zeros((self.n_lines, self.n_voxel_batch), device=self.dev)
        concentration_map_rot_batch_flat = concentration_map_rot_batch.view(self.n_element, self.n_voxel_batch)
        line_idx = 0
        for j in range(self.n_element):
            ## for step 1
            lac_single = concentration_map_rot_batch[j] * self.probe_attCS_ls[j]
            lac_acc = tc.cumsum(lac_single, axis=1)
            lac_acc = tc.cat((tc.zeros((self.minibatch_size, 1), device=self.dev), lac_acc), dim = 1)
            att_exponent_acc = lac_acc * (self.sample_size_cm / self.sample_size_n)    
            att_exponent_acc_map += att_exponent_acc
            
            ## for step 2
            fl_unit = self.detected_fl_unit_concentration[line_idx:line_idx + self.n_line_group_each_element[j]]            
            ## FL signal over the current elemental lines for each voxel
            fl_map = tc.stack([concentration_map_rot_batch_flat[j] * fl_unit_single_line for fl_unit_single_line in fl_unit])            
            fl_map_tot_flat_theta[line_idx:line_idx + self.n_line_group_each_element[j],:] = fl_map            
            line_idx = line_idx + len(fl_unit)
            
        attenuation_map_theta_flat = tc.exp(-(att_exponent_acc_map[:,:-1])).view(self.n_voxel_batch)
        transmission_theta = tc.exp(-att_exponent_acc_map[:,-1])
        
            
        ### 3: Calculate SA (the map of attenuation ratio due to self-absorption of the FL signal):
        # 1. for each FL emitting source voxel (n_voxel_batch),
        # 2. we have an attenuation ratio when the photon for each elemental line reaches the edge of the sample (due to self absorption) (n_lines)
        # ==> The dimension of SA: (n_lines, n_voxel_batch)
              
        # First, Calaulting the exponent of attenuation: att_exponent
        # lac: linear attenuation coefficient = concentration * attenuation_cross_section, dimension: (n_element, n_lines, n_voxel_batch(FL source), n_voxel)  
           # For each voxel source in this batch, there're n_lines emitted XRF energy, each voxel in the sample contribute to the attenuation due to SA
        # FL_line_attCS_ls: an arrary of the attenuation cross section, dimension: (n_element, n_lines)
           # The component in the array represents the total attenuation cross section at some line energy in some element
               
        lac = concentration_map_rot.view(self.n_element, 1, 1, self.n_voxel) * self.FL_line_attCS_ls.view(self.n_element, self.n_lines, 1, 1)
        lac = lac.expand(-1, -1, self.n_voxel_batch, -1).float()
        
        voxel_idx_offset = self.p * self.n_voxel_batch
        
        att_exponent = tc.stack([lac[:,:, tc.clamp((self.P_batch[m,0] - voxel_idx_offset), 0, self.n_voxel_batch).to(dtype=tc.long), self.P_batch[m,1].to(dtype=tc.long)]
                                 * self.P_batch[m,2].view(1, 1, -1).repeat(self.n_element, self.n_lines, 1) for m in range(self.n_det)])
        
        ## summing over the attenation exponent contributed by all intersecting voxels, dim = (n_det, n_element, n_lines, n_voxel_batch(FL source))
        att_exponent_voxel_sum = tc.sum(att_exponent.view(self.n_det, self.n_element, self.n_lines, self.n_voxel_batch, self.dia_len_n), axis=-1)
        
        ## calculate the attenuation caused by all elements, dim = (n_det, n_lines, n_voxel_batch(FL source)), and then take the average over n_det FL paths
        SA_theta =  tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0)        
        
              
        #### 4: Create XRF, XRT data ####           
        probe_after_attenuation_theta = self.probe_before_attenuation_flat * attenuation_map_theta_flat 
        fl_signal_SA_theta = tc.unsqueeze(probe_after_attenuation_theta, dim=0) * fl_map_tot_flat_theta * SA_theta  
        fl_signal_SA_theta = fl_signal_SA_theta.view(-1, self.minibatch_size, self.sample_size_n)
        fl_signal_SA_theta = tc.sum(fl_signal_SA_theta, axis=-1)
                       
        output1 = fl_signal_SA_theta
        output2 = self.probe_cts * transmission_theta
#         print("running_time = %.3f" %(time.time() - start_time))
        return output1, output2



class PPM_cont(nn.Module):
    
    fl_line_groups=np.array(["K", "L", "M"]) 
    fl_K=fl_K
    fl_L=fl_L
    fl_M=fl_M
    group_lines=True
    
    def __init__(self, dev, grid_concentration, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                 this_aN_dic, probe_energy, probe_cts, fl_line_groups, fl_K, fl_L, fl_M, group_lines, 
                 theta_st, theta_end, n_theta, this_theta_idx,
                 det_ds_spacing_cm, det_size_cm, det_from_sample_cm, P_save_path):
        """
        Initialize the attributes of PPM. 
        """
        super(PPM_cont, self).__init__() # inherit the __init__ from nn.Module.
        self.dev = dev
        self.grid_concentration = grid_concentration
        self.theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1]
        self.this_theta_idx = this_theta_idx
        self.n_element = n_element
        self.sample_height_n = sample_height_n
        self.minibatch_size = minibatch_size
        self.sample_size_n = sample_size_n
        self.p = p  # indicate which minibatch to calculate the gradient  
        self.xp = self.init_xp() # initialize the values of the minibatch
        
        self.probe_energy = probe_energy  
        self.this_aN_dic = this_aN_dic
        self.n_element = tc.as_tensor(len(self.this_aN_dic)).to(self.dev)
        self.element_ls = np.array(list(this_aN_dic.keys()))
        self.aN_ls = np.array(list(this_aN_dic.values()))
        
        self.probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.probe_energy).flatten()).to(self.dev)
#         self.probe_attCS_dic = dict(zip(self.element_ls, self.probe_attCS_ls))
        
        self.sample_size_cm = sample_size_cm
        self.fl_line_groups = fl_line_groups
        self.fl_K = fl_K
        self.fl_L = fl_L
        self.fl_M = fl_M
        self.group_lines = group_lines
        self.fl_all_lines_dic = self.init_fl_all_lines_dic()
        self.n_lines = tc.as_tensor(self.fl_all_lines_dic["n_lines"]).to(self.dev)
        self.FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.fl_all_lines_dic["fl_energy"])).float().to(self.dev)
        self.detected_fl_unit_concentration = tc.as_tensor(self.fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(self.dev)
        self.n_line_group_each_element = tc.IntTensor(self.fl_all_lines_dic["n_line_group_each_element"]).to(self.dev)
        
        self.dia_len_n = int((self.sample_height_n**2 + self.sample_size_n**2 + self.sample_size_n**2)**0.5)
        self.n_voxel_batch = self.minibatch_size * self.sample_size_n
        self.n_voxel = self.sample_height_n * self.sample_size_n**2   
        
        self.det_ds_spacing_cm = det_ds_spacing_cm
        self.det_size_cm = det_size_cm
        self.det_from_sample_cm = det_from_sample_cm
        self.P_save_path = P_save_path
        
        self.n_det, self.P_batch = self.init_intersecting_length_fl_detectorlet()  
        
        self.probe_cts = probe_cts
        self.probe_before_attenuation_flat = self.init_probe()

        
    def init_xp(self):
        """
        Initialize self.x with the tensor of the saved intermediate reconstructing results (n_element, minibatch_size, n_y)
        """
        theta = self.theta_ls[self.this_theta_idx]
        concentration_map_rot = rotate(self.grid_concentration, theta, self.dev).view(self.n_element, self.sample_height_n * self.sample_size_n, self.sample_size_n)
        return nn.Parameter(concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :])

    def init_fl_all_lines_dic(self):
        """
        Initialize self.fl_all_lines_dic
        """
        fl_all_lines_dic = MakeFLlinesDictionary(self.this_aN_dic, self.probe_energy,
                      self.sample_size_n.cpu().numpy(), self.sample_size_cm.cpu().numpy(),
                      self.fl_line_groups, self.fl_K, self.fl_L, self.fl_M,
                      self.group_lines)
        return fl_all_lines_dic
    
    def init_intersecting_length_fl_detectorlet(self):
        """
        Initialize self.intersecting_length_fl_detectorlet
        """
        n_det, P = intersecting_length_fl_detectorlet_3d(self.det_size_cm, self.det_from_sample_cm, self.det_ds_spacing_cm,
                                                  self.sample_size_n.cpu().numpy(), self.sample_size_cm.cpu().numpy(),
                                                  self.sample_height_n.cpu().numpy(), self.P_save_path)
   
        P = P.float().to(self.dev)       
        P = P.view(n_det, 3, self.dia_len_n * self.sample_height_n * self.sample_size_n, self.sample_size_n)
        P_batch = P[:, :, self.dia_len_n * self.minibatch_size * self.p : self.dia_len_n * self.minibatch_size * (self.p+1), :].detach().clone()
        P_batch = P_batch.view(n_det, 3, self.dia_len_n * self.minibatch_size * self.sample_size_n)
        del P

        return n_det, P_batch
    
    def init_probe(self):
        return self.probe_cts * tc.ones((self.minibatch_size * self.sample_size_n), device=self.dev)
    
    def forward(self, grid_concentration, this_theta_idx): 
        """
        Forward propagation.
        """      
        
        ### 1: Calculate the map of attenuation and transmission ###          
        theta = self.theta_ls[this_theta_idx]

        # Rotate the sample
        concentration_map_rot = rotate(grid_concentration, theta, self.dev).view(self.n_element, self.sample_height_n * self.sample_size_n, self.sample_size_n)
        
        # Set part of the sample to be the updating target
        concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :] = self.xp
        
        # Create a tensor to store the current batch of the sample
        concentration_map_rot_batch = concentration_map_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :]
        
        ## Calculate the attenuation of the probe
        # Calculate the expoenent of attenuation of each voxel in the batch. (The atteuation before the probe enters each voxel.)
        att_exponent_acc_map = tc.zeros((self.minibatch_size, self.sample_size_n+1), device=self.dev)
        
        fl_map_tot_flat_theta = tc.zeros((self.n_lines, self.n_voxel_batch), device=self.dev)
        concentration_map_rot_batch_flat = concentration_map_rot_batch.view(self.n_element, self.n_voxel_batch)
        line_idx = 0
        for j in range(self.n_element):
            ## for step 1
            lac_single = concentration_map_rot_batch[j] * self.probe_attCS_ls[j]
            lac_acc = tc.cumsum(lac_single, axis=1)
            lac_acc = tc.cat((tc.zeros((self.minibatch_size, 1), device=self.dev), lac_acc), dim = 1)
            att_exponent_acc = lac_acc * (self.sample_size_cm / self.sample_size_n)    
            att_exponent_acc_map += att_exponent_acc
            
            ## for step 2
            fl_unit = self.detected_fl_unit_concentration[line_idx:line_idx + self.n_line_group_each_element[j]]            
            ## FL signal over the current elemental lines for each voxel
            fl_map = tc.stack([concentration_map_rot_batch_flat[j] * fl_unit_single_line for fl_unit_single_line in fl_unit])            
            fl_map_tot_flat_theta[line_idx:line_idx + self.n_line_group_each_element[j],:] = fl_map            
            line_idx = line_idx + len(fl_unit)
            
        attenuation_map_theta_flat = tc.exp(-(att_exponent_acc_map[:,:-1])).view(self.n_voxel_batch)
        transmission_theta = tc.exp(-att_exponent_acc_map[:,-1])
        
            
        ### 3: Calculate SA (the map of attenuation ratio due to self-absorption of the FL signal):
        # 1. for each FL emitting source voxel (n_voxel_batch),
        # 2. we have an attenuation ratio when the photon for each elemental line reaches the edge of the sample (due to self absorption) (n_lines)
        # ==> The dimension of SA: (n_lines, n_voxel_batch)
        
        
        # First, Calaulting the exponent of attenuation: att_exponent
        # lac: linear attenuation coefficient = concentration * attenuation_cross_section, dimension: (n_element, n_lines, n_voxel_batch(FL source), n_voxel)  
           # For each voxel source in this batch, there're n_lines emitted XRF energy, each voxel in the sample contribute to the attenuation due to SA
        # FL_line_attCS_ls: an arrary of the attenuation cross section, dimension: (n_element, n_lines)
           # The component in the array represents the total attenuation cross section at some line energy in some element
               
        lac = concentration_map_rot.view(self.n_element, 1, 1, self.n_voxel) * self.FL_line_attCS_ls.view(self.n_element, self.n_lines, 1, 1)
        lac = lac.expand(-1, -1, self.n_voxel_batch, -1).float()
        
        voxel_idx_offset = self.p * self.n_voxel_batch
        
        att_exponent = tc.stack([lac[:,:, tc.clamp((self.P_batch[m,0] - voxel_idx_offset), 0, self.n_voxel_batch).to(dtype=tc.long), self.P_batch[m,1].to(dtype=tc.long)]
                                 * self.P_batch[m,2].view(1, 1, -1).repeat(self.n_element, self.n_lines, 1) for m in range(self.n_det)])
        
        ## summing over the attenation exponent contributed by all intersecting voxels, dim = (n_det, n_element, n_lines, n_voxel_batch(FL source))
        att_exponent_voxel_sum = tc.sum(att_exponent.view(self.n_det, self.n_element, self.n_lines, self.n_voxel_batch, self.dia_len_n), axis=-1)
        
        ## calculate the attenuation caused by all elements, dim = (n_det, n_lines, n_voxel_batch(FL source)), and then take the average over n_det FL paths
        SA_theta =  tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0)        
        
              
        #### 4: Create XRF, XRT data ####           
        probe_after_attenuation_theta = self.probe_before_attenuation_flat * attenuation_map_theta_flat 
        fl_signal_SA_theta = tc.unsqueeze(probe_after_attenuation_theta, dim=0) * fl_map_tot_flat_theta * SA_theta  
        fl_signal_SA_theta = fl_signal_SA_theta.view(-1, self.minibatch_size, self.sample_size_n)
        fl_signal_SA_theta = tc.sum(fl_signal_SA_theta, axis=-1)
                       
        output1 = fl_signal_SA_theta
        output2 = self.probe_cts * transmission_theta

        return output1, output2