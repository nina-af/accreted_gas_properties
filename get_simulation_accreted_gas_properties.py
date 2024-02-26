import numpy as np
import sys
import os

sys.path.append('/work/08381/nina_af/ls6/accreted_gas_properties')

from sink_accretion_history import Cloud, SnapshotGasProperties, SinkAccretionHistory

bhdir     = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/blackhole_details/'
snapdir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
datadir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/accreted_gas_properties/'
acc_dict  = SinkAccretionHistory(bhdir).accretion_dict

M0, R0, alpha0 = 2e4, 3.0, 2.0
cloud = Cloud(M0, R0, alpha0)

def get_fname(i, snapdir=snapdir):
    return os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))

# Set snapshot range (later: do automatically?)
i_min, i_max = 0, 10

# Loop over snapshots.
for i in range(i_min, i_max, 1):

    s = SnapshotGasProperties(get_fname(i), cloud)
    all_data = s.get_all_gas_data(acc_dict)
    s.write_to_file(all_data, acc_dict, datadir)
