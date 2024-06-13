#!/usr/bin/env python3

import numpy as np
import h5py as h5
import sys
import os

sys.path.append('/work/08381/nina_af/ls6/accreted_gas_properties')

from sink_accretion_history import SinkAccretionHistory
from sink_particle_systems import read_star_snapshot, FindHierarchicalSystems

bhdir    = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/blackhole_details/'
snapdir  = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
datadir  = '/scratch/08381/nina_af/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/accreted_gas_properties/'
acc_dict = SinkAccretionHistory(bhdir).accretion_dict
all_ids  = acc_dict['sink_IDs']

# Label sink particles in snapshot.
def get_labels(i, snapdir=snapdir, all_ids=all_ids, stars_only=False, verbose=False):
    m, x, v, ids = read_star_snapshot(i, snapdir, stars_only=stars_only)
    systems      = FindHierarchicalSystems(m, x, v, ids, max_system_pop=4); systems.find_multiples()
    labels       = -1 * np.ones(len(all_ids))
    for i, sink_id in enumerate(all_ids):
        exists_in_snap = np.any(ids == sink_id)
        if not exists_in_snap:
            labels[i] = 0
            if verbose:
                print('{0:d} not in snapshot.'.format(sink_id))
        else:
            j = np.where(systems.systems == sink_id)[0][0]
            n = np.sum(systems.systems[j, :] >= [0, 0, 0, 0])
            if verbose:
                print('{0:d} in row {1:d}\tsystem size = {2:d}'.format(sink_id, j, n))
            labels[i] = n
    #labels_int = labels.astype(int)
    return labels

# Set snapshot range (later: pass as argument/detect from snapshot directory?)
i_min, i_max = 0, 750
all_data     = np.zeros((i_max+1 - i_min, len(all_ids)))

# Loop over snapshots.
for i in range(i_min, i_max+1, 1):
    labels         = get_labels(i)
    all_data[i, :] = labels

# Write to HDF5 file: all sink IDs; sink labels for all snapshots.
fname = os.path.join(datadir, 'sink_system_labels.hdf5')
f     = h5.File(fname, 'w')
f.create_dataset('sink_IDs', data=np.asarray(all_ids), dtype=int)
f.create_dataset('labels', data=all_data)
f.close()
