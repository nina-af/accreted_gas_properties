#!/usr/bin/env python3

import os
import glob
import numpy as np
import h5py
import pytreegrav as pg

class SinkAccretionHistory:

    """
    Class for getting particle IDs and accretion times for all gas particles to 
    be accreted onto a sink particle for all sink particles formed in a 
    STARFORGE simulation.
    Needs bhswallow_%d.txt, bhformation_%d.txt files in blackhole_details.
    Adapted from A. Kaalva.
    """
    
    # Initialize with path to blackhole_details directory (bhdir).
    def __init__(self, bhdir):
        self.bhdir          = bhdir
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

        # (Optionally) save as text file.
        if save_txt:
            # If no filename, save in blackhole_details directory.
            if fname is None:
                fname = os.path.join(self.bhdir, 'sink_accretion_data.txt')
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
    
        # (Optionally) save as text file.
        if save_txt:
            # If no filename, save in blackhole_details directory.
            if fname is None:
                fname = os.path.join(self.bhdir, 'sink_formation_data.txt')
            fmt   = '%d','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g','%.8g'  
            np.savetxt(fname, np.vstack((sink_IDs, t, m, x, y, z, u, v, w)).T, fmt=fmt)
            
        return sink_IDs, sink_formation_data
    
    # Parse accretion_data, sink_formation_data to return dict containing
    # gas IDs, accretion times, formation properties for each sink particle.
    # dict keys: sink IDs (int_arr)
    # dict values:
    #   - accretion_dict[sink_ID][0] = [accreted_gas_ids (int_arr), accretion_times (float_arr)]
    #   - accretion_dict[sink_ID][1] = sink [t, m, x, y, z, vx, vy, vz] at formation.
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
    
        accretion_dict = dict.fromkeys(unique_sinks.astype(np.int64))
        for i in range(len(unique_sinks)):
            accretion_dict[unique_sinks[i]] = [gas_IDs_split[i].astype(np.int64), acc_times_split[i]]
        accretion_dict['sink_IDs'] = unique_sinks.astype(np.int64)
    
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

class Cloud:
    """
    Class for calculating bulk cloud properties.
    Parameters:
        - M0: initial cloud mass [code_mass].
        - R0: initial cloud radius [code_length].
        - alpha0: initial cloud turbulent virial parameter.
        - G_code: gravitational constant in code units
        (default: 4300.17 in [pc * Msun^-1 * (m/s)^2]).
    """

    def __init__(self, M, R, alpha, G_code=4300.71, verbose=False):

        # Initial cloud mass, radius, and turbulent virial parameter.
        # G_code: gravitational constant in code units [default: 4300.71].
        self.M = M
        self.R = R
        self.L = (4.0 * np.pi * self.R**3 / 3.0)**(1.0/3.0)
        self.alpha  = alpha
        self.G_code = G_code

        self.rho     = self.get_initial_density(verbose=verbose)
        self.Sigma   = self.get_initial_surface_density(verbose=verbose)
        self.vrms    = self.get_initial_sigma_3D(verbose=verbose)
        self.t_cross = self.get_initial_t_cross(verbose=verbose)
        self.t_ff    = self.get_initial_t_ff(verbose=verbose)

    # ----------------------------- FUNCTIONS ---------------------------------

    # FROM INITIAL CLOUD PARAMETERS: surface density, R, vrms, Mach number.
    def get_initial_density(self, verbose=False):
        """
        Calculate the initial cloud density [code_mass/code_length**3].
        """
        rho = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        if verbose:
            print('Density: {0:.2f} Msun pc^-2'.format(rho))
        return rho


    def get_initial_surface_density(self, verbose=False):
        """
        Calculate the initial cloud surface density [code_mass/code_length**2].
        """
        Sigma = self.M / (np.pi * self.R**2)
        if verbose:
            print('Surface density: {0:.2f} Msun pc^-2'.format(Sigma))
        return Sigma

    # Initial 3D rms velocity.
    def get_initial_sigma_3D(self, verbose=False):
        """
        Calculate the initial 3D rms velocity [code_velocity].
        """
        sig_3D = np.sqrt((3.0 * self.alpha * self.G_code * self.M) / (5.0 * self.R))
        if verbose:
            print('sigma_3D = {0:.3f} m s^-1'.format(sig_3D))
        return sig_3D

    # Initial cloud trubulent crossing time.
    def get_initial_t_cross(self, verbose=False):
        """
        Calculate the initial turbulent crossing time as R/v_rms [code_time].
        """
        sig_3D  = self.get_initial_sigma_3D()
        t_cross = self.R / sig_3D
        if verbose:
            print('t_cross = {0:.3g} [code]'.format(t_cross))
        return t_cross

    # Initial cloud gravitational free-fall time.
    def get_initial_t_ff(self, verbose=False):
        """
        Calculate the initial freefall time [code_time].
        """
        rho  = (3.0 * self.M) / (4.0 * np.pi * self.R**3)
        t_ff = np.sqrt((3.0 * np.pi) / (32.0 * self.G_code * rho))
        if verbose:
            print('t_ff = {0:.3g} [code]'.format(t_ff))
        return t_ff

class SnapshotGasProperties:
    """
    Read gas particle data from HDF5 snapshot.
    """

    def __init__(self, fname, cloud, B_unit=1e4, res_limit=1e-3):

        # Physical constants.
        self.PROTONMASS_CGS     = 1.6726e-24
        self.ELECTRONMASS_CGS   = 9.10953e-28
        self.BOLTZMANN_CGS      = 1.38066e-16
        self.HYDROGEN_MASSFRAC  = 0.76
        self.ELECTRONCHARGE_CGS = 4.8032e-10
        self.C_LIGHT_CGS        = 2.9979e10

        # Initial cloud parameters.
        self.fname   = fname
        self.snapdir = self.get_snapdir()
        self.Cloud   = cloud
        self.M0      = cloud.M      # Initial cloud mass, radius.
        self.R0      = cloud.R
        self.L0      = cloud.L      # Volume-equivalent length.
        self.alpha0  = cloud.alpha  # Initial virial parameter.
        
        # Resolution limit for masking feedback cells.
        self.res_limit = res_limit

        # Open HDF5 file.
        with h5py.File(fname, 'r') as f:
            header = f['Header']
            p0     = f['PartType0']   # Particle type 0 (gas).

            # Header attributes.
            self.box_size = header.attrs['BoxSize']
            self.num_p0   = header.attrs['NumPart_Total'][0]
            self.t        = header.attrs['Time']

            # Unit conversions to cgs; note typo in header for G_code.
            self.G_code      = header.attrs['Gravitational_Constant_In_Code_Inits']
            if 'Internal_UnitB_In_Gauss' in header.attrs:
                self.B_code = header.attrs['Internal_UnitB_In_Gauss']  # sqrt(4*pi*unit_pressure_cgs)
            else:
                self.B_code = 2.916731267922059e-09
            self.l_unit      = header.attrs['UnitLength_In_CGS']
            self.m_unit      = header.attrs['UnitMass_In_CGS']
            self.v_unit      = header.attrs['UnitVelocity_In_CGS']
            self.B_unit      = B_unit                           # Magnetic field unit in Gauss (default: 1e4).
            self.t_unit      = self.l_unit / self.v_unit
            self.t_unit_myr  = self.t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
            self.rho_unit    = self.m_unit / self.l_unit**3
            self.nH_unit     = self.rho_unit/self.PROTONMASS_CGS
            self.P_unit      = self.m_unit / self.l_unit / self.t_unit**2
            self.spec_L_unit = self.l_unit * self.v_unit        # Specific angular momentum.
            self.L_unit      = self.spec_L_unit * self.m_unit   # Angular momentum.
            self.E_unit      = self.l_unit**2 / self.t_unit**2  # Energy per mass.
            # Convert internal energy to temperature units.
            self.u_to_temp_units = (self.PROTONMASS_CGS/self.BOLTZMANN_CGS)*self.E_unit

            # Other useful conversion factors.
            self.cm_to_AU = 6.6845871226706e-14
            self.cm_to_pc = 3.2407792896664e-19

            # PartType0 data.
            self.p0_ids   = p0['ParticleIDs'][()]         # Particle IDs.
            self.p0_m     = p0['Masses'][()]              # Masses.
            self.p0_rho   = p0['Density'][()]             # Density.
            self.p0_hsml  = p0['SmoothingLength'][()]     # Particle smoothing length.
            self.p0_E_int = p0['InternalEnergy'][()]      # Internal energy.
            self.p0_P     = p0['Pressure'][()]            # Pressure.
            self.p0_cs    = p0['SoundSpeed'][()]          # Sound speed.
            self.p0_x     = p0['Coordinates'][()][:, 0]   # Coordinates.
            self.p0_y     = p0['Coordinates'][()][:, 1]
            self.p0_z     = p0['Coordinates'][()][:, 2]
            self.p0_u     = p0['Velocities'][()][:, 0]    # Velocities.
            self.p0_v     = p0['Velocities'][()][:, 1]
            self.p0_w     = p0['Velocities'][()][:, 2]
            self.p0_Ne    = p0['ElectronAbundance'][()]   # Electron abundance.
            if 'MagneticField' in p0.keys():              # Magnetic field.
                self.p0_Bx    = p0['MagneticField'][()][:, 0]
                self.p0_By    = p0['MagneticField'][()][:, 1]
                self.p0_Bz    = p0['MagneticField'][()][:, 2]
                self.p0_B_mag = np.sqrt(self.p0_Bx**2 + self.p0_By**2 + self.p0_Bz**2)
            else:
                self.p0_Bx    = np.zeros(len(self.p0_ids))
                self.p0_By    = np.zeros(len(self.p0_ids))
                self.p0_Bz    = np.zeros(len(self.p0_ids))
                self.p0_B_mag = np.zeros(len(self.p0_ids))

            # Hydrogen number density and total metallicity.
            self.p0_n_H  = (1.0 / self.PROTONMASS_CGS) * \
                       np.multiply(self.p0_rho * self.rho_unit, 1.0 - p0['Metallicity'][()][:, 0])
            self.p0_total_metallicity = p0['Metallicity'][()][:, 0]
            # Calculate mean molecular weight.
            self.p0_mean_molecular_weight = self.get_mean_molecular_weight(self.p0_ids)

            # Neutral hydrogen abundance, molecular mass fraction.
            self.p0_neutral_H_abundance = p0['NeutralHydrogenAbundance'][()]
            self.p0_molecular_mass_frac = p0['MolecularMassFraction'][()]

            # Calculate gas adiabatic index and temperature.
            fH, f, xe            = self.HYDROGEN_MASSFRAC, self.p0_molecular_mass_frac, self.p0_Ne
            f_mono, f_di         = fH*(xe + 1.-f) + (1.-fH)/4., fH*f/2.
            gamma_mono, gamma_di = 5./3., 7./5.
            gamma                = 1. + (f_mono + f_di) / (f_mono/(gamma_mono-1.) + f_di/(gamma_di-1.))
            self.p0_temperature  = (gamma - 1.) * self.p0_mean_molecular_weight * \
                                    self.u_to_temp_units * self.p0_E_int
            # Dust temperature.
            if 'Dust_Temperature' in p0.keys():
                self.p0_dust_temp = p0['Dust_Temperature'][()]

            # Simulation timestep.
            if 'TimeStep' in p0.keys():
                self.p0_timestep = p0['TimeStep'][()]

            # For convenience, stack coordinates/velocities/B_field in a (n_gas, 3) array.
            self.p0_coord = np.vstack((self.p0_x, self.p0_y, self.p0_z)).T
            self.p0_vel   = np.vstack((self.p0_u, self.p0_v, self.p0_w)).T
            self.p0_mag   = np.vstack((self.p0_Bx, self.p0_By, self.p0_Bz)).T
            
    # Try to get snapshot datadir from filename.
    def get_snapdir(self):
        return self.fname.split('snapshot_')[0]

    # Calculate gas mean molecular weight.
    def get_mean_molecular_weight(self, gas_ids):
        idx_g                 = np.isin(self.p0_ids, gas_ids)
        T_eff_atomic          = 1.23 * (5.0/3.0-1.0) * self.u_to_temp_units * self.p0_E_int[idx_g]
        nH_cgs                = self.p0_rho[idx_g] * self.nH_unit
        T_transition          = self._DMIN(8000., nH_cgs)
        f_mol                 = 1./(1. + T_eff_atomic**2/T_transition**2)
        return 4. / (1. + (3. + 4.*self.p0_Ne[idx_g] - 2.*f_mol) * self.HYDROGEN_MASSFRAC)
            
    # Check that mask based on gas_ids, res_limit is non_zero.
    def check_gas_ids(self, gas_ids, verbose=True):
        mask_ids = np.isin(self.p0_ids, gas_ids)
        mask_res = (self.p0_m >= 0.99 * self.res_limit)
        mask_all = np.logical_and(mask_ids, mask_res)
        
        if np.sum(mask_all) > 0:
            return True
        else:
            if verbose:
                print('No gas_ids above resolution limit.')
            return False
            
    # Get mask based on gas_ids, res_limit.
    def get_mask(self, gas_ids):
        mask_ids = np.isin(self.p0_ids, gas_ids)
        mask_res = (self.p0_m >= 0.99 * self.res_limit)
        mask_all = np.logical_and(mask_ids, mask_res)
        
        return mask_all
        
    # Get indices of selected gas particles.
    def get_idx(self, gas_ids):
        if np.isscalar(gas_ids):
            idx_g = np.where(self.p0_ids == gas_ids)[0][0]
        else:
            idx_g = np.isin(self.p0_ids, gas_ids)
        return idx_g
        
    # Get center of mass for selected gas particles.
    def get_center_of_mass(self, gas_ids):
        idx_g   = self.get_idx(gas_ids)
        m, M    = self.p0_m[idx_g], np.sum(self.p0_m[idx_g])
        x, y, z = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        u, v, w = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]

        x_cm = np.sum(np.multiply(m, x))/M; u_cm = np.sum(np.multiply(m, u))/M
        y_cm = np.sum(np.multiply(m, y))/M; v_cm = np.sum(np.multiply(m, v))/M
        z_cm = np.sum(np.multiply(m, z))/M; w_cm = np.sum(np.multiply(m, w))/M

        cm_x = np.asarray([x_cm, y_cm, z_cm])
        cm_v = np.asarray([u_cm, v_cm, w_cm])

        return M, cm_x, cm_v
        
    # Get gas kinematics relative to x, v vectors.
    def get_relative_kinematics(self, gas_ids, point_x, point_v):
        x0, y0, z0 = point_x[0], point_x[1], point_x[2]
        u0, v0, w0 = point_v[0], point_v[1], point_v[2]
        
        idx_g   = self.get_idx(gas_ids)
        m       = self.p0_m[idx_g]
        x, y, z = self.p0_x[idx_g] - x0, self.p0_y[idx_g] - y0, self.p0_z[idx_g] - z0
        u, v, w = self.p0_u[idx_g] - u0, self.p0_v[idx_g] - v0, self.p0_w[idx_g] - w0
        
        if np.isscalar(gas_ids):
            return m, np.asarray([x, y, z]), np.asarray([u, v, w])
        else:
            return m, np.vstack((x, y, z)).T, np.vstack((u, v, w)).T
            
    # Get total mass of remaining gas.
    def get_total_mass(self, gas_ids):
        idx_g = self.get_idx(gas_ids)
        m     = self.p0_m[idx_g]
        return np.sum(m)
        
    # Get effective radius of selected gas particles.
    def get_effective_radius(self, gas_ids):
        idx_g = self.get_idx(gas_ids)
        m     = self.p0_m[idx_g]
        rho   = self.p0_rho[idx_g]
        vol   = np.sum(m/rho)
        r     = ((3.0 * vol) / (4.0 * np.pi))**(1.0/3.0)
        return r
        
    # Get velocity dispersion of selected gas particles.
    def get_velocity_dispersion(self, gas_ids):
        idx_g   = self.get_idx(gas_ids)
        m       = self.p0_m[idx_g]
        u, v, w = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]
        sigma_3D = np.sqrt((self.weight_std(u, m)**2.0 + self.weight_std(v, m)**2.0 + \
                            self.weight_std(w, m)**2.0))
        return sigma_3D
        
    # Get angular momentum (with respect to center of mass) of selected gas particles.
    def get_angular_momentum(self, gas_ids):
        m_cm, x_cm, v_cm = self.get_center_of_mass(gas_ids)
        m_g, x_g, v_g    = self.get_relative_kinematics(gas_ids, x_cm, v_cm)
        ang_mom_vec      = np.sum(np.cross(x_g, np.multiply(np.reshape(m_g, (len(m_g), 1)), v_g)), axis=0)
        ang_mom_mag      = np.linalg.norm(ang_mom_vec)
        ang_mom_unit_vec = ang_mom_vec / ang_mom_mag
        return ang_mom_unit_vec, ang_mom_mag

    # Get gravitational potential energy (need pytreegrav module).
    def get_potential_energy(self, gas_ids):
        idx_g   = self.get_idx(gas_ids)
        m, h    = self.p0_m[idx_g], self.p0_hsml[idx_g]
        x, y, z = self.p0_x[idx_g], self.p0_y[idx_g], self.p0_z[idx_g]
        pos     = np.vstack((x, y, z)).T
        E_pot   = 0.5 * np.sum(m * pg.Potential(pos, m, h, G=self.G_code))
        return E_pot
        
    # Get kinetic energy [code units].
    def get_kinetic_energy(self, gas_ids):
        idx_g   = self.get_idx(gas_ids)
        m       = self.p0_m[idx_g]
        u, v, w = self.p0_u[idx_g], self.p0_v[idx_g], self.p0_w[idx_g]
        vel     = np.vstack((u, v, w)).T
        dv      = vel - np.average(vel, weights=m, axis=0)
        v_sqr   = np.sum(dv**2,axis=1)
        E_kin   = 0.5 * np.sum(m * v_sqr)
        return E_kin
        
    # Get magnetic energy [code units].
    def get_magnetic_energy(self, gas_ids):
        idx_g   = self.get_idx(gas_ids)
        m, rho  = self.p0_m[idx_g], self.p0_rho[idx_g]
        B_mag   = self.p0_B_mag[idx_g] * self.B_unit
        vol     = (m / rho) * self.l_unit**3
        E_mag   = (1.0/(8.0 * np.pi)) * np.sum(B_mag*B_mag*vol) / (self.E_unit * self.m_unit)
        return E_mag
    

    # Utility functions.
    def weight_avg(self, data, weights):
        "Weighted average"
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sum(data * weights) / weightsum
        else:
            return 0
    def weight_std(self, data, weights):
        "Weighted standard deviation."
        weights   = np.abs(weights)
        weightsum = np.sum(weights)
        if (weightsum > 0):
            return np.sqrt(np.sum(((data - self.weight_avg(data, weights))**2) * weights) / weightsum)
        else:
            return 0
    def _sigmoid_sqrt(self, x):
        return 0.5*(1 + x/np.sqrt(1+x*x))
    def _DMIN(self, a, b):
        return np.where(a < b, a, b)
    def _DMAX(self, a, b):
        return np.where(a > b, a, b)
