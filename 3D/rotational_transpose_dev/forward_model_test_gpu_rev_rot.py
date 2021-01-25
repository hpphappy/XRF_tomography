import numpy as np
import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import xraylib as xlib
import xraylib_np as xlib_np
import torch.nn as nn
from data_generation_fns_rev_rot import rotate, MakeFLlinesDictionary, intersecting_length_fl_detectorlet_3d


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
    
    def __init__(self, dev, lac, ini_kind, init_const, grid_concentration, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                 this_aN_dic, probe_energy, probe_cts,
                 theta_st, theta_end, n_theta, this_theta_idx,
                 n_det, P_batch):
        """
        Initialize the attributes of PPM. 
        """
        super(PPM, self).__init__() # inherit the __init__ from nn.Module.
        self.dev = dev
        self.lac = lac.to(dev)
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

        self.fl_all_lines_dic = self.init_fl_all_lines_dic()
        self.n_lines = tc.as_tensor(self.fl_all_lines_dic["n_lines"]).to(self.dev)
        self.FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.fl_all_lines_dic["fl_energy"])).float().to(self.dev)
        self.detected_fl_unit_concentration = tc.as_tensor(self.fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(self.dev)
        self.n_line_group_each_element = tc.IntTensor(self.fl_all_lines_dic["n_line_group_each_element"]).to(self.dev)
        
        self.dia_len_n = int((self.sample_height_n**2 + self.sample_size_n**2 + self.sample_size_n**2)**0.5)
        self.n_voxel_minibatch = self.minibatch_size * self.sample_size_n
        self.n_voxel = self.sample_height_n * self.sample_size_n**2   
        
        self.n_det = n_det
        self.P_batch = P_batch.to(dev)     
        self.SA_theta = self.init_SA_theta()
        
        self.probe_cts = probe_cts
        self.probe_before_attenuation_flat = self.init_probe()

        
    def init_xp(self):
        """
        Initialize self.xp with the tensor of the true model(noise may already be added) (n_element, minibatch_size, n_y)
        """
        if self.ini_kind == "rand" or self.ini_kind == "randn":
            theta = self.theta_ls[self.this_theta_idx]
            cmap_rot = rotate(self.grid_concentration, theta, self.dev).view(self.n_element, self.sample_height_n * self.sample_size_n, self.sample_size_n)
            cmap_rot_this_minibatch = cmap_rot[:, self.minibatch_size * self.p : self.minibatch_size*(self.p+1), :]
            cmap_rot_this_minibatch = cmap_rot_this_minibatch.view(self.n_element, self.minibatch_size//self.sample_size_n, self.sample_size_n, self.sample_size_n)
            
            return nn.Parameter(cmap_rot_this_minibatch)
        
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
    

    def init_SA_theta(self): 
        voxel_idx_offset = self.p * self.n_voxel_minibatch
        att_exponent = tc.stack([self.lac[:,:, tc.clamp((self.P_batch[m,0] - voxel_idx_offset), 0, self.n_voxel_minibatch).to(dtype=tc.long), self.P_batch[m,1].to(dtype=tc.long)]
                                 * self.P_batch[m,2].view(1, 1, -1).repeat(self.n_element, self.n_lines, 1) for m in range(self.n_det)])
        
        ## summing over the attenation exponent contributed by all intersecting voxels, dim = (n_det, n_element, n_lines, n_voxel_minibatch(FL source))
        att_exponent_voxel_sum = tc.sum(att_exponent.view(self.n_det, self.n_element, self.n_lines, self.n_voxel_minibatch, self.dia_len_n), axis=-1)
        
        ## calculate the attenuation caused by all elements, dim = (n_det, n_lines, n_voxel_minibatch(FL source)), and then take the average over n_det FL paths
        ## The final dimension of SA_theta = (n_lines, n_voxel_minibatch(FL source))
        SA_theta =  tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0) 
        
        return SA_theta

    
    def init_probe(self):
        return self.probe_cts * tc.ones((self.minibatch_size * self.sample_size_n), device=self.dev)
    
    def forward(self): 
        """
        Forward propagation.
        """      
        
        ### 1: Calculate the map of attenuation and transmission ###          
        
        # Create a tensor to store the current updated part of the sample
        concentration_map_rot_batch = self.xp
        
        ## Calculate the attenuation of the probe
        ## Calculate the expoenent of attenuation of each voxel in the batch. (The atteuation before the probe enters each voxel.)
        att_exponent_acc_map = tc.zeros((self.minibatch_size, self.sample_size_n+1), device=self.dev)
        
        fl_map_tot_flat_theta = tc.zeros((self.n_lines, self.n_voxel_minibatch), device=self.dev)
        concentration_map_rot_batch_flat = concentration_map_rot_batch.view(self.n_element, self.n_voxel_minibatch)
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
            
        attenuation_map_theta_flat = tc.exp(-(att_exponent_acc_map[:,:-1])).view(self.n_voxel_minibatch)
        transmission_theta = tc.exp(-att_exponent_acc_map[:,-1])
        
              
        #### 4: Create XRF, XRT data ####           
        probe_after_attenuation_theta = self.probe_before_attenuation_flat * attenuation_map_theta_flat 
        fl_signal_SA_theta = tc.unsqueeze(probe_after_attenuation_theta, dim=0) * fl_map_tot_flat_theta * self.SA_theta  
        fl_signal_SA_theta = fl_signal_SA_theta.view(-1, self.minibatch_size, self.sample_size_n)
        fl_signal_SA_theta = tc.sum(fl_signal_SA_theta, axis=-1)
                       
        output1 = fl_signal_SA_theta
        output2 = self.probe_cts * transmission_theta

        return output1, output2



class PPM_cont(nn.Module):
    
    fl_line_groups=np.array(["K", "L", "M"]) 
    fl_K=fl_K
    fl_L=fl_L
    fl_M=fl_M
    group_lines=True
    
    def __init__(self, dev, lac, xp, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                 this_aN_dic, probe_energy, probe_cts,
                 theta_st, theta_end, n_theta, this_theta_idx,
                 n_det, P_batch):
        """
        Initialize the attributes of PPM. 
        """
        super(PPM_cont, self).__init__() # inherit the __init__ from nn.Module.
        self.dev = dev
        self.lac = lac.to(self.dev)
        self.theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1]
        self.this_theta_idx = this_theta_idx
        self.n_element = n_element
        self.sample_height_n = sample_height_n
        self.minibatch_size = minibatch_size
        self.sample_size_n = sample_size_n
        self.p = p  # indicate which minibatch to calculate the gradient  
        self.xp = xp # initial values of the minibatch, dimension = [C, N(this minibatch), H, W]
        
        self.probe_energy = probe_energy  
        self.this_aN_dic = this_aN_dic
        self.n_element = tc.as_tensor(len(self.this_aN_dic)).to(self.dev)
        self.element_ls = np.array(list(this_aN_dic.keys()))
        self.aN_ls = np.array(list(this_aN_dic.values()))
        
        self.probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.probe_energy).flatten()).to(self.dev)
        
        self.sample_size_cm = sample_size_cm
        self.fl_all_lines_dic = self.init_fl_all_lines_dic()
        self.n_lines = tc.as_tensor(self.fl_all_lines_dic["n_lines"]).to(self.dev)
        self.FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(self.aN_ls, self.fl_all_lines_dic["fl_energy"])).float().to(self.dev)
        self.detected_fl_unit_concentration = tc.as_tensor(self.fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(self.dev)
        self.n_line_group_each_element = tc.IntTensor(self.fl_all_lines_dic["n_line_group_each_element"]).to(self.dev)
        
        self.dia_len_n = int((self.sample_height_n**2 + self.sample_size_n**2 + self.sample_size_n**2)**0.5)
        self.n_voxel_minibatch = self.minibatch_size * self.sample_size_n
        self.n_voxel = self.sample_height_n * self.sample_size_n**2   
        
        self.n_det = n_det
        self.P_batch = P_batch.to(dev)     
        self.SA_theta = self.init_SA_theta()
        
        self.probe_cts = probe_cts
        self.probe_before_attenuation_flat = self.init_probe()
     

    def init_fl_all_lines_dic(self):
        """
        Initialize self.fl_all_lines_dic
        """
        fl_all_lines_dic = MakeFLlinesDictionary(self.this_aN_dic, self.probe_energy,
                      self.sample_size_n.cpu().numpy(), self.sample_size_cm.cpu().numpy(),
                      self.fl_line_groups, self.fl_K, self.fl_L, self.fl_M,
                      self.group_lines)
        return fl_all_lines_dic
       

    def init_SA_theta(self): 
        voxel_idx_offset = self.p * self.n_voxel_minibatch
        att_exponent = tc.stack([self.lac[:,:, tc.clamp((self.P_batch[m,0] - voxel_idx_offset), 0, self.n_voxel_minibatch).to(dtype=tc.long), self.P_batch[m,1].to(dtype=tc.long)]
                                 * self.P_batch[m,2].view(1, 1, -1).repeat(self.n_element, self.n_lines, 1) for m in range(self.n_det)])
        
        ## summing over the attenation exponent contributed by all intersecting voxels, dim = (n_det, n_element, n_lines, n_voxel_minibatch(FL source))
        att_exponent_voxel_sum = tc.sum(att_exponent.view(self.n_det, self.n_element, self.n_lines, self.n_voxel_minibatch, self.dia_len_n), axis=-1)
        
        ## calculate the attenuation caused by all elements, dim = (n_det, n_lines, n_voxel_minibatch(FL source)), and then take the average over n_det FL paths
        SA_theta =  tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0) 
        
        return SA_theta

    def init_probe(self):
        return self.probe_cts * tc.ones((self.n_voxel_minibatch), device=self.dev)
    
    def forward(self): 
        """
        Forward propagation.
        """      
        
        ### 1: Calculate the map of attenuation and transmission ###  
        
        # Create a tensor to store the current updated part of the sample
        concentration_map_rot_batch = self.xp ## dimension = [C, N(this minibatch), H, W]
        concentration_map_rot_batch = tc.reshape(concentration_map_rot_batch, (self.n_element, self.minibatch_size, self.sample_size_n))
        ## Calculate the attenuation of the probe
        # Calculate the expoenent of attenuation of each voxel in the batch. (The atteuation before the probe enters each voxel.)
        att_exponent_acc_map = tc.zeros((self.minibatch_size, self.sample_size_n+1), device=self.dev)
        
        fl_map_tot_flat_theta = tc.zeros((self.n_lines, self.n_voxel_minibatch), device=self.dev)
        concentration_map_rot_batch_flat = concentration_map_rot_batch.view(self.n_element, self.n_voxel_minibatch)
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
            
        attenuation_map_theta_flat = tc.exp(-(att_exponent_acc_map[:,:-1])).view(self.n_voxel_minibatch)
        transmission_theta = tc.exp(-att_exponent_acc_map[:,-1])
        
              
        #### 4: Create XRF, XRT data ####  
        # calculate probe intensity at each voxel, dim = [n_voxel_minibatch]
        probe_after_attenuation_theta = self.probe_before_attenuation_flat * attenuation_map_theta_flat
        
        # calculate signal emmitted from each source voxel in this minibatch after going through self-absorption, dim = [n_voxel_minibatch]
        fl_signal_SA_theta = tc.unsqueeze(probe_after_attenuation_theta, dim=0) * fl_map_tot_flat_theta * self.SA_theta  
        
        # fold the dimension back to [minibatch_size, sample_size_n]
        fl_signal_SA_theta = fl_signal_SA_theta.view(-1, self.minibatch_size, self.sample_size_n)
        
        # summing over the dimension "sample_size_n" to calculate the siganl collected from each strip[beam position]
        fl_signal_SA_theta = tc.sum(fl_signal_SA_theta, axis=-1) 
                       
        output1 = fl_signal_SA_theta
        output2 = self.probe_cts * transmission_theta

        return output1, output2
    
    