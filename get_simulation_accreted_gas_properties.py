#!/usr/bin/env python3

import numpy as np
import sys
import os

sys.path.append('/work/08381/nina_af/frontera/accreted_gas_properties')

from sink_accretion_history_speedup import Cloud, SnapshotGasProperties, SinkAccretionHistory

#bhdir     = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/blackhole_details/'
#snapdir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
#datadir   = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/accreted_gas_properties_v2/'

bhdir     = '/scratch1/08381/nina_af/multiplicity/blackhole_details_fiducial_42/'
snapdir   = '/scratch3/03532/mgrudic/STARFORGE_RT/STARFORGE_v1.1/M2e4_R10/M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42/output/'
datadir   = '/scratch1/08381/nina_af/multiplicity/data_fiducial_42/'

fname_gas  = os.path.join(bhdir, 'sink_accretion_data.txt')
fname_sink = os.path.join(bhdir, 'sink_formation_data.txt')

print('Getting accretion dict...')
acc_dict  = SinkAccretionHistory(bhdir, fname_gas=fname_gas, fname_sink=fname_sink).accretion_dict

M0, R0, alpha0 = 2e4, 10.0, 2.0
cloud = Cloud(M0, R0, alpha0)

def get_fname(i, snapdir=snapdir):
    return os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))

# Set snapshot range (later: pass as argument/detect from snapshot directory?)
i_min, i_max = 0, 454  # fiducial_42

# Set sink particle range (split into batches of ~20 sink particles).
sink_imin, sink_imax = 0, 99

# Loop over snapshots.
for i in range(i_min, i_max+1, 1):

    print('Writing snapshot {0:d}...'.format(i), flush=True)
    s = SnapshotGasProperties(get_fname(i), cloud)
    all_data = s.get_all_gas_data(acc_dict, use_all_sinks=False, sink_imin=sink_imin, sink_imax=sink_imax)
    s.write_to_file(all_data, datadir, use_all_sinks=False, sink_imin=sink_imin, sink_imax=sink_imax)
