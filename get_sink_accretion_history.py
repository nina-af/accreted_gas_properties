#!/usr/bin/env python3

import numpy as np
import sys
import os
import h5py

sys.path.append('/work/08381/nina_af/frontera/accreted_gas_properties')

from sink_accretion_history import SinkAccretionHistory

#bhdir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/blackhole_details/'
#datadir = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/accreted_gas_properties_v2/'

#bhdir   = '/scratch1/08381/nina_af/M50_n5e6_nonideal_reruns/output_nonideal_OAH/blackhole_details/'
#datadir = '/scratch1/08381/nina_af/M50_n5e6_nonideal_reruns/output_nonideal_OAH/accreted_gas_properties/'

bhdir   = '/scratch1/08381/nina_af/M50_n5e6_nonideal_reruns/output_nonideal_OAH_restart_v1/blackhole_details/'
datadir = '/scratch1/08381/nina_af/M50_n5e6_nonideal_reruns/output_nonideal_OAH_restart_v1/accreted_gas_properties/'

acc_dict  = SinkAccretionHistory(bhdir).accretion_dict

sink_IDs  = acc_dict['sink_IDs']
num_sinks = len(sink_IDs)

# Snapshot range.
#imin, imax = 291, 366   # nonideal_OAH
imin, imax = 311, 337   # nonideal_OAH_restart_v1

# Initialize [num_snapshots x num_data]-dim array for each sink ID;
# loop over snapshots and add data to row i for each sink ID.

# data_empty = np.zeros((imax - imin, 26))
data_empty = np.zeros((imax+1 - imin, 25))    # Remove M_inc field.
sink_data_dict = dict.fromkeys(sink_IDs.astype(np.int64))

for key in sink_data_dict.keys():
    sink_data_dict[key] = np.copy(data_empty)

m_unit, l_unit, v_unit, t_unit, B_unit = 0.0, 0.0, 0.0, 0.0, 0.0

for i in range(imin, imax+1):
    
    fname = os.path.join(datadir, 'snapshot_{0:03d}_accreted_gas_properties.hdf5'.format(i))
    
    with h5py.File(fname, 'r') as f:
        header = f['header']
        time   = header.attrs['time']
        m_unit = header.attrs['m_unit']
        l_unit = header.attrs['l_unit']
        v_unit = header.attrs['v_unit']
        t_unit = header.attrs['t_unit']
        B_unit = header.attrs['B_unit']
        
        M_tot  = f['M_tot'][()]
        x_cm   = f['X_cm'][()]
        v_cm   = f['V_cm'][()]
        #L_vec  = f['angular_momentum'][()]
        L_vec  = f['specific_angular_momentum'][()]
        R_eff  = f['effective_radius'][()]
        R_p    = f['aspect_ratio'][()]
        T      = f['temperature'][()]
        B      = f['magnetic_field_strength'][()]
        Ne     = f['electron_abundance'][()]
        sig3D  = f['velocity_dispersion'][()]
        E_grav = f['potential_energy'][()]
        E_kin  = f['kinetic_energy'][()]
        E_mag  = f['magnetic_energy'][()]
        E_int  = f['internal_energy'][()]
        N_inc  = f['included_particle_number'][()]
        N_fb   = f['feedback_particle_number'][()]
        #M_fb   = f['feedback_particle_mass'][()]
    
    for j, sink_ID in enumerate(sink_IDs):
        data = sink_data_dict[sink_ID]

        data[i-imin, 0]     = M_tot[j]
        data[i-imin, 1:4]   = x_cm[j]
        data[i-imin, 4:7]   = v_cm[j]
        data[i-imin, 7:10]  = L_vec[j]
        data[i-imin, 10]    = R_eff[j]
        data[i-imin, 11:14] = R_p[j]
        data[i-imin, 14]    = T[j]
        data[i-imin, 15]    = B[j]
        data[i-imin, 16]    = Ne[j]
        data[i-imin, 17]    = sig3D[j]
        data[i-imin, 18]    = E_grav[j]
        data[i-imin, 19]    = E_kin[j]
        data[i-imin, 20]    = E_mag[j]
        data[i-imin, 21]    = E_int[j]
        data[i-imin, 22]    = N_inc[j]
        data[i-imin, 23]    = N_fb[j]
        #data[i, 24]    = M_fb[j]
        data[i-imin, 24]    = time

        sink_data_dict[sink_ID] = data
        
# Save each dict as an HDF5 dataset.
fname  = os.path.join(datadir, 'all_accreted_gas_properties_nonideal_OAH_restart_v1.hdf5')
f      = h5py.File(fname, 'w')
header = f.create_dataset('header', (1,))
header.attrs.create('m_unit', m_unit, dtype=float)
header.attrs.create('l_unit', l_unit, dtype=float)
header.attrs.create('v_unit', v_unit, dtype=float)
header.attrs.create('t_unit', t_unit, dtype=float)
header.attrs.create('B_unit', B_unit, dtype=float)

for sink_ID in sink_IDs:
    f.create_dataset('sink_{0:d}'.format(sink_ID), data=sink_data_dict[sink_ID])
    
f.close()
