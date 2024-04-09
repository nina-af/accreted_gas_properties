#!/usr/bin/env python3

import os
import numpy as np
import h5py

# Get masses, coordinates, velocities, particle IDs from HDF5 snapshots.
def read_star_snapshot(i, snapdir, stars_only=True):
    if stars_only:
        fname = os.path.join(snapdir, 'snapshot_{0:03d}_stars.hdf5'.format(i))
    else:
        fname = os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))

    with h5py.File(fname, 'r') as f:
        header = f['Header']

        # If no PartType5 data yet, return None for all fields.
        if not 'PartType5' in f:
            return None, None, None, None
        p5     = f['PartType5']

        p5_ids = p5['ParticleIDs'][()]           # Particle IDs.
        p5_m   = p5['Masses'][()]                # Masses.
        p5_x   = p5['Coordinates'][()][:, 0]     # Coordinates.
        p5_y   = p5['Coordinates'][()][:, 1]
        p5_z   = p5['Coordinates'][()][:, 2]
        p5_u   = p5['Velocities'][()][:, 0]      # Velocities.
        p5_v   = p5['Velocities'][()][:, 1]
        p5_w   = p5['Velocities'][()][:, 2]

    x = np.vstack((p5_x, p5_y, p5_z)).T
    v = np.vstack((p5_u, p5_v, p5_w)).T
    return p5_m, x, v, p5_ids

class FindHierarchicalSystems:

    def __init__(self, m, x, v, ids, max_system_pop=4):
        """
        'Calculate relative energy between point like bodies and find most
        bound pairs and systems, like in Bate 2009'
        Adapted from D. Guszejnov.
        """
        self.m   = np.copy(m)
        self.x   = np.copy(x)
        self.v   = np.copy(v)
        self.ids = np.copy(ids)

        #Constants
        self.pc     = 3.08567758e+16  # m
        self.msolar = 1.98892e+30     # kg
        self.Gconst = 6.67428e-11     # mks system

        self.max_system_pop = max_system_pop
        self.nstar          = m.size

        self.fill_energy_matrix()

        # Start with each star in its own system.
        self.systems       = np.zeros([self.nstar, max_system_pop], dtype=np.int64) - 1
        self.systems[:, 0] = ids
        self.make_systempops()
        self.id_matrix     = np.zeros([self.nstar, self.nstar])
        for i in range(self.nstar):
            self.id_matrix[i,:] = ids

    def make_systempops(self):
        self.systempops = np.sum(self.systems>=0, axis=1)

    def fill_energy_matrix(self):
        factor = self.Gconst * self.msolar / self.pc;
        self.energy_matrix = np.zeros([self.nstar, self.nstar])
        for i in range(self.nstar):
            for j in range(i+1, self.nstar):
                mtot    = self.m[i] + self.m[j]
                mu      = self.m[i] * self.m[j] / mtot
                dx      = self.x[i, :] - self.x[j, :]
                dv_vect = self.v[i, :] - self.v[j, :]
                dr      = np.sqrt(np.sum(dx * dx))
                dvsq    = np.sum(dv_vect * dv_vect)
                self.energy_matrix[i, j] = 0.5 * dvsq * mu - factor * mtot * mu / dr
        #self.energy_matrix *= self.msolar # comments indicate this is extra

    def rebuild_energy_matrix(self,ind1,ind2):
        factor = self.Gconst * self.msolar / self.pc
        self.energy_matrix[:, ind2] = 0
        self.energy_matrix[ind2, :] = 0
        for i in range(self.nstar):
                for j in range(i+1, self.nstar):
                    if ((i==ind1) or (j==ind1) or (i==ind2) or (j==ind2)):
                        if ((self.m[j]==0) or (self.m[i]==0)):
                            self.energy_matrix[i, j] = 0
                        else:
                            mtot    = self.m[i] + self.m[j]
                            mu      = self.m[i] * self.m[j] / mtot
                            dx      = self.x[i, :] - self.x[j, :]
                            dv_vect = self.v[i, :] - self.v[j, :]
                            dr      = np.sqrt(np.sum(dx * dx))
                            dvsq    = np.sum(dv_vect * dv_vect)
                            self.energy_matrix[i,j] = 0.5 * dvsq * mu - factor * mtot * mu / dr

    def find_min_position(matrix):
        return divmod(matrix.argmin(), matrix.shape[1])

    def make_energy_list(self):
        self.energylist = self.energy_matrix.flatten()
        self.sortind    = self.energylist.argsort()
        # Sort in increasing order.
        self.energylist.sort()
        self.id1list = (self.id_matrix.flatten())[self.sortind]
        self.id2list = ((self.id_matrix.transpose()).flatten())[self.sortind]

    def merge_systems(self, ind1, ind2):
        # Populations.
        self.systems[ind1, self.systempops[ind1]:(self.systempops[ind1] + self.systempops[ind2])] = \
        self.systems[ind2, 0:self.systempops[ind2]]
        self.systems[ind2,:] = -1
        self.make_systempops()
        # Center of mass position, velocity, mass of merged system.
        self.x[ind1, :] = (self.x[ind1, :] * self.m[ind1] + self.x[ind2, :] * self.m[ind2]) / (self.m[ind1] + self.m[ind2])
        self.v[ind1, :] = (self.v[ind1, :] * self.m[ind1] + self.v[ind2, :] * self.m[ind2]) / (self.m[ind1] + self.m[ind2])
        self.m[ind1]    = self.m[ind1] + self.m[ind2]
        self.m[ind2]    = 0

    def find_multiples(self):
        changed = 1
        while(changed):
            self.make_energy_list()
            i       = 0
            changed = 0
            # Iterate over bound systems.
            while(self.energylist[i] < 0):
                pos  = np.squeeze(np.argwhere(self.energy_matrix==self.energylist[i]))
                ind1 = pos[0]
                ind2 = pos[1]
                # Merge systems if resulting population size is less than max_system_pop.
                if((self.systempops[ind1] + self.systempops[ind2]) <= self.max_system_pop):
                    self.merge_systems(ind1, ind2)
                    changed = 1
                    break
                i += 1
            # Recalculate the energy list.
            if(changed):
                self.rebuild_energy_matrix(ind1, ind2);

    def find_binaries(self):
        changed = 1
        while changed:
            self.make_energy_list();
            i       = 0
            changed = 0
            # Iterate over bound systems.
            while(self.energylist[i] < 0):
                pos  = np.squeeze(np.argwhere(self.energy_matrix==self.energylist[i]))
                ind1 = pos[0]
                ind2 = pos[1]
                # Merge systems if binary.
                if((self.systempops[ind1] + self.systempops[ind2]) <= 2):
                    self.merge_systems(ind1, ind2)
                    changed = 1
                    break
                i += 1
            # Recalculate the energy list.
            if(changed):
                self.rebuild_energy_matrix(ind1,ind2);


class StarSystemInfo:
    """
    Get information about a single star system, e.g. single line in
    FindHierarchicalSystems.systems.
    Adapted from D. Guszejnov.
    """

    def __init__(self, m, x, v, ids, idlist):

        # Unit definitions (TO-DO: get code_units)
        self.pc     = 3.08567758e+16
        self.msolar = 1.9891e+30
        self.Gconst = 6.67384e-11
        self.AU     = 149597870700.0
        self.yr     = 31556952.0

        index    = np.squeeze(np.nonzero(np.isin(ids, idlist)))
        self.pop = np.sum(idlist>-1)

        self.semimajor_matrix = np.zeros([self.pop, self.pop])
        self.t_orbital_matrix = np.zeros([self.pop, self.pop])
        self.energy_matrix    = np.zeros([self.pop, self.pop])
        if (m.size > 1):
            self.m = m[index]
        else:
            self.m = np.array([m])

        # Sort by mass.
        if (self.pop > 1):
            sortind = (-self.m).argsort()
            index   = index[sortind]
            self.m   = m[index]
            self.x   = x[index, :]
            self.v   = v[index, :]
            self.ids = ids[index]
            self.energy_matrix = (FindHierarchicalSystems(self.m, self.x, self.v, self.ids)).energy_matrix
            # Symmetrize.
            self.energy_matrix = self.energy_matrix + np.transpose(self.energy_matrix)
            m_tot_matrix       = np.zeros([self.pop, self.pop])
            m_red_matrix       = np.zeros([self.pop, self.pop])
            for i in range(self.pop):
                for j in range(self.pop):
                    m_tot_matrix[i, j] = self.m[i] + self.m[j]
                    m_red_matrix[i, j] = self.m[i] * self.m[j] / m_tot_matrix[i, j]
            self.semimajor_matrix = np.zeros([self.pop, self.pop])
            self.t_orbital_matrix = np.zeros([self.pop, self.pop])
            energy_ind = (self.energy_matrix < 0)
            self.semimajor_matrix[energy_ind] =-(self.msolar**2)*self.Gconst*m_tot_matrix[energy_ind] \
                                                /(2.0*self.energy_matrix[energy_ind]/m_red_matrix[energy_ind])
            self.t_orbital_matrix[energy_ind] = 2.0*np.pi*np.sqrt((self.semimajor_matrix[energy_ind]**3.0) \
                                                /(self.Gconst*self.msolar*m_tot_matrix[energy_ind]))
            # Convert to units.
            self.semimajor_matrix /= self.pc
            self.t_orbital_matrix /= self.yr
            # Take only the two most massive stars in the system.
            self.semimajor = self.semimajor_matrix[0, 1]
            self.t_orbital = self.t_orbital_matrix[0, 1]
        else:
            if (m.size>1):
                self.m   = m[index]
                self.x   = x[index, :]
                self.v   = v[index, :]
                self.ids = ids[index]
            else:
                self.m   = m
                self.x   = x
                self.v   = v
                self.ids = ids
            self.semimajor = 0.0
            self.t_orbital = 0.0
        if (self.m.size == 1):
            self.m   = np.array([self.m])
            self.x   = np.array([self.x])
            self.v   = np.array([self.v])
            self.ids = np.array([self.ids])
        self.m_tot = np.sum(self.m)
        self.xCoM = np.sum(self.m[:, None] * self.x, axis=0) / self.m_tot
        self.vCoM = np.sum(self.m[:, None] * self.v, axis=0) / self.m_tot
