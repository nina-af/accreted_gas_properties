#!/usr/bin/env python3

import os
import glob
import numpy as np
import h5py

class SinkAccretionHistory:

    """
    Class for getting particle IDs and accretion times for all gas particles to 
    be accreted onto a sink particle for all sink particles formed in a 
    STARFORGE simulation.
    Needs bhswallow_%d.txt, bhformation_%d.txt files in blackhole_details.
    Adapted from A. Kaalva.
    """
    
    def __init__(self, bhdir, datadir, label):
        self.bhdir   = bhdir      # location of blackhole_details directory.
        self.datadir = datadir    # store generated .txt files, etc.
        self.label   = label      # simulation label, e.g., M2e3_R3_a2..., for filenames.
        
        self.accretion_dict = self.accretion_history_to_dict()
        
    def get_accretion_history(self, fname=None, save_txt=False):
    
        # List of bhswallow_%d.txt filenames:
        bhswallow_files = glob.glob(self.bhdir + 'bhswallow*.txt')
    
        # Read all lines from all bhswallow files:
        bhswallow_total = []
        for i in range(len(bhswallow_files)):
            with open(bhswallow_files[i]) as f:
                data_list = f.read().splitlines()
                for j in range(len(data_list)):
                    new_data = data_list[j].split()
                    bhswallow_total.append(new_data)
                
        # Lists of sink IDs, gas IDs, and accretion times from bhswallow_data:
        sink_ID_list = []
        gas_ID_list  = []
        time_list    = []
        for i in range(len(bhswallow_total)):
            time_list.append(float(bhswallow_total[i][0]))
            sink_ID_list.append(int(bhswallow_total[i][1]))
            gas_ID_list.append(int(bhswallow_total[i][6]))
        sink_IDs = np.asarray(sink_ID_list, dtype=int)
        gas_IDs  = np.asarray(gas_ID_list, dtype=int)
        times    = np.asarray(time_list, dtype=float)
        
        # Sort by sink ID to group accretion history for each sink particle.
        idx_sink_sort    = np.argsort(sink_IDs)
        sink_IDs_grouped = sink_IDs[idx_sink_sort]
        gas_IDs_grouped  = gas_IDs[idx_sink_sort]
        times_grouped    = times[idx_sink_sort]
    
        # Get first occurrence of each unique sink ID in grouped list:
        unique_sinks, first_occurrence = np.unique(sink_IDs_grouped, return_index=True)
    
        # Within each sink particle group, sort accreted gas particles by accretion time.
        idx_times_sort = []
        for i in range(len(first_occurrence) - 1):
            imin = first_occurrence[i]
            imax = first_occurrence[i+1]
            # Sort times[imin:imax]
            idx_t = np.argsort(times_grouped[imin:imax]) + imin
            idx_times_sort.extend(idx_t.tolist())
        idx_t = np.argsort(times_grouped[imax:]) + imax
        idx_times_sort.extend(idx_t.tolist())

        sink_IDs_sort = sink_IDs_grouped[idx_times_sort]
        gas_IDs_sort  = gas_IDs_grouped[idx_times_sort]
        times_sort    = times_grouped[idx_times_sort]
    
        # Stack into (N_gas_accreted, 3) array to save to text file.
        accretion_data = np.vstack((sink_IDs_sort, gas_IDs_sort, times_sort)).T
    
        # Write to text file.
        if save_txt:
            if fname is None:
                fname = os.path.join(self.datadir, 
                                     ('accretion_data_' + self.label + '.txt'))
            fmt = '%d', '%d', '%.8f'
            np.savetxt(fname, accretion_data, fmt=fmt)
        
        return accretion_data

    # Get sink particle properties at formation time.
    def get_sink_formation_details(self, fname=None, save_txt=False):
    
        # List of bhformation_%d.txt filenames:
        bhformation_files = glob.glob(self.bhdir + 'bhformation*.txt')
    
        # Read all lines from all bhswallow files:
        bhformation_total = []
        for i in range(len(bhformation_files)):
            with open(bhformation_files[i]) as f:
                data_list = f.read().splitlines()
                for j in range(len(data_list)):
                    new_data = data_list[j].split()
                    bhformation_total.append(new_data)
                
        # Lists of sink IDs, formation time, mass, position, velocity
        sink_ID_list = []
        m_list, t_list = [], []
        x_list, y_list, z_list = [], [], []
        u_list, v_list, w_list = [], [], []

        for i in range(len(bhformation_total)):
            sink_ID_list.append(int(bhformation_total[i][1]))
            t_list.append(float(bhformation_total[i][0]))
            m_list.append(float(bhformation_total[i][2]))
            x_list.append(float(bhformation_total[i][3]))
            y_list.append(float(bhformation_total[i][4]))
            z_list.append(float(bhformation_total[i][5]))
            u_list.append(float(bhformation_total[i][6]))
            v_list.append(float(bhformation_total[i][7]))
            w_list.append(float(bhformation_total[i][8]))
    
        sink_IDs = np.asarray(sink_ID_list, dtype=int)
        t        = np.asarray(t_list, dtype=float)
        m        = np.asarray(m_list, dtype=float)
        x        = np.asarray(x_list, dtype=float)
        y        = np.asarray(y_list, dtype=float)
        z        = np.asarray(z_list, dtype=float)
        u        = np.asarray(u_list, dtype=float)
        v        = np.asarray(v_list, dtype=float)
        w        = np.asarray(w_list, dtype=float)

        sink_formation_data = np.vstack((t, m, x, y, z, u, v, w)).T
    
        # Write to text file.
        if save_txt:
            if fname is None:
                fname = os.path.join(self.datadir, 
                                     ('sink_formation_data_' + self.label + '.txt'))
            fmt   = '%d','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g'  
            np.savetxt(fname, np.vstack((sink_IDs, t, m, x, y, z, u, v, w)).T, fmt=fmt)
            
        return sink_IDs, sink_formation_data
    
    # Parse accretion_data, sink_formation_data text files to return dict of 
    # sink ID: gas IDs, accretion times, formation properties for each sink particle.
    def accretion_history_to_dict(self, fname_gas=None, fname_sink=None):
        
        # Read accretion_data from stored .txt file.
        if fname_gas is not None:
            sink_IDs  = np.loadtxt(fname_gas, dtype=int, usecols=0)
            gas_IDs   = np.loadtxt(fname_gas, dtype=int, usecols=1)
            acc_times = np.loadtxt(fname_gas, dtype=float, usecols=2)
        else:
            accretion_data = self.get_accretion_history()
            sink_IDs  = accretion_data[:, 0]
            gas_IDs   = accretion_data[:, 1]
            acc_times = accretion_data[:, 2]
    
        unique_sinks, first_occurrence = np.unique(sink_IDs, return_index=True)
    
        # Split gas IDs, accretion times into list of subarrays per sink particle.
        gas_IDs_split   = np.split(gas_IDs, first_occurrence[1:])
        acc_times_split = np.split(acc_times, first_occurrence[1:])
    
        accretion_dict = dict.fromkeys(unique_sinks)
        for i in range(len(unique_sinks)):
            accretion_dict[unique_sinks[i]] = [gas_IDs_split[i], acc_times_split[i]]
        accretion_dict['sink_IDs'] = unique_sinks
    
        # Add sink particle properties at formation time to dict.
        if fname_sink is not None:
            s_IDs  = np.loadtxt(fname_sink, dtype=int, usecols=0)
            s_data = np.loadtxt(fname_sink, dtype=float, usecols=(1, 2, 3, 4, 5, 6, 7, 8))
        else:
            s_IDs, s_data = self.get_sink_formation_details()
            
        for i, sink_ID in enumerate(s_IDs):
            if sink_ID in accretion_dict:
                accretion_dict[sink_ID] = [accretion_dict[sink_ID], s_data[i]]
    
        return accretion_dict
 
# TO-DO: want to sort sink particles in snapshot into singles, binaries, higher-order
# systems - add "system type" to above accretion_dict?
        
        
