import json

import numpy as np
from fact.analysis.statistics import li_ma_significance
from read_mars import read_mars
from scipy.optimize import minimize_scalar
from blockspec.spec.calc_a_eff_parallel import calc_a_eff_parallel_hd5
from blockspec.spec.plotting import plot_spectrum, plot_theta
from blockspec.spec.read_data import histos_from_list_of_mars_files, calc_onoffhisto
from tqdm import tqdm

from blockspec.spec.on_time_parallel import calc_on_time


def symmetric_log10_errors(value, error):
    """ Calculate upper and lower error, that appear symmetric in loglog-plots.

    :param value: ndarray or float
    :param error: ndarray or float
    :return: array of lower error and upper error.
    """
    error /= (value * np.log(10))
    error_l = value - np.ma.power(10, (np.ma.log10(value) - error))
    error_h = np.ma.power(10, (np.ma.log10(value) + error)) - value
    return np.ma.array([error_l, error_h])


def save_variables_to_json(self, filename):
    data = {}
    for variable_name in self.list_of_variables:
        data[variable_name] = getattr(self, variable_name)

    for entry in data:
        if isinstance(data[entry], (np.ndarray, np.ma.core.MaskedArray)):
            aslist = data[entry].tolist()
            data[entry] = aslist
        elif isinstance(data[entry], dict):
            for element in data[entry]:
                aslist = data[entry][element]
                if isinstance(data[entry][element], (np.ndarray, np.ma.core.MaskedArray)):
                    aslist = data[entry][element].tolist()
                data[entry][element] = aslist
        elif isinstance(data[entry], list):
            for id, element in enumerate(data[entry]):
                if isinstance(element, dict):
                    for thing in element:
                        aslist = data[entry][id][thing]
                        if isinstance(aslist, (np.ndarray, np.ma.core.MaskedArray)):
                            aslist = aslist.tolist()
                            data[entry][id][thing] = aslist
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_variables_from_json(self, filename):
    with open(filename) as infile:
        data = json.load(infile)

    for variable_name in data:
        containing = data[variable_name]
        if variable_name in self.list_of_variables:
            if isinstance(containing, list):
                if len(containing) > 0:
                    if isinstance(containing[0], dict):
                        pass
                    else:
                        containing = np.array(containing)
                        if None in containing:  # allows to load masked numpy arrays
                            containing = np.ma.masked_invalid(containing.astype(np.float))
            setattr(self, variable_name, containing)

        else:
            print("Key", variable_name, "not in list of variables.")
            if variable_name in ["energy_labels", "zenith_labels", "ll_dicts"]:  # Keep them for backwards compatibility
                pass
            else:
                raise KeyError('Key not in list of variables')


class Spectrum:
    """ Class containing FACT spectra and additional information"""
    list_of_variables = ["use_correction_factors",
                         "theta_square",
                         "alpha",
                         "list_of_ceres_files",
                         "ganymed_file_mc",
                         "run_list_star",
                         "ganymed_file_data",
                         "energy_center",
                         "energy_error",
                         "energy_binning",
                         "zenith_binning",
                         "on_time_per_zd",              # 1D-Array of On-Time per ZenithDistance bin
                         "total_on_time",               # Total On-Time of the observation
                         "zenith_binning_on_time",      # Save the zenith binning used for On-Time calculation
                         "on_histo_zenith",             # 2D-Histogram Energy:ZenithDistance of On-Events
                         "off_histo_zenith",            # 2D-Histogram Energy:ZenithDistance of Off-Events
                         "on_histo",                    # 1D-Histogram in Energy of On-Events
                         "off_histo",                   # 1D-Histogram in Energy of Off-Events
                         "significance_histo",          # 1D-Histogram in Energy of Significance
                         "excess_histo",                # 1D-Histogram in Energy of Excess_Events
                         "excess_histo_err",            # 1D-Histogram in Energy of Error of Excess_Events
                         "n_on_events",                 # Total number (sum) of On-Events
                         "n_off_events",                # Total number (sum) of Off-Events
                         "n_excess_events",             # Total number (sum) of Excess-Events
                         "n_excess_events_err",         # Estimated error of total number of Excess_events
                         "overall_significance",        # Overall Significance (computed with total On- and Off-events)
                         "theta_square_binning",        # 1D-Histogram of binning in theta_square
                         "on_theta_square_histo",       # 1D-Histogram in Theta Square of On-Events
                         "off_theta_square_histo",      # 1D-Histogram in Theta Square of Off-Events
                         "effective_area",              # 2D-Histogram in Energy:ZenithDistance of Effective Area
                         "effective_area_err",          # 2D-Histogram in Energy:ZenithDistance of Error of A_eff
                         "scaled_effective_area",       # effective_area scaled by On-Time per zenith bin
                         "scaled_effective_area_err",   # scaled error of effective area error by On-Time per zenith bin
                         "energy_migration",     # 3D-Histogram in EnergyMC:Energy:ZenithDistance of surviving MC events
                         "scaled_energy_migration",     # energy migration scaled by On-Time per zenith bin
                         "differential_spectrum",       # 1D-Histogram, in energy of spectral points dN/dE
                         "differential_spectrum_err",   # 1D-Histogram, estimated error of spectral points

                         # Dict containing overall stats: number of on, off and excess events, 
                         # total on-time in hours, significance:
                         "stats"]

    def __init__(self,
                 run_list_star=None,
                 theta_sq=0.085,
                 correction_factors=False,
                 ebins=None,
                 zdbins=None,
                 ganymed_file_data=None,
                 ganymed_file_mc=None,
                 list_of_ceres_files=None,
                 alpha=0.2
                 ):

        self.use_correction_factors = correction_factors
        self.theta_square = theta_sq
        self.alpha = alpha

        self.list_of_ceres_files = list_of_ceres_files
        self.ganymed_file_mc = ganymed_file_mc

        if run_list_star:
            self.run_list_star = run_list_star

        if ebins:
            self.energy_binning = ebins
        else:
            self.energy_binning = np.logspace(np.log10(200.0), np.log10(50000.0), 9)
        if zdbins:
            self.zenith_binning = zdbins
        else:
            self.zenith_binning = np.linspace(0, 60, 15)

        self.ganymed_file_data = ganymed_file_data

        self.energy_center = None
        self.energy_error = None

        # Declare Placeholder variables
        self.on_time_per_zd = None
        self.total_on_time = None
        self.zenith_binning_on_time = None

        self.on_histo_zenith = None
        self.off_histo_zenith = None
        self.on_histo = None
        self.off_histo = None
        self.significance_histo = None
        self.excess_histo = None
        self.excess_histo_err = None
        self.n_on_events = None
        self.n_off_events = None
        self.n_excess_events = None
        self.n_excess_events_err = None
        self.overall_significance = None

        self.theta_square_binning = None
        self.on_theta_square_histo = None
        self.off_theta_square_histo = None

        self.effective_area = None
        self.effective_area_err = None
        self.scaled_effective_area = None
        self.scaled_effective_area_err = None
        self.energy_migration = None
        self.scaled_energy_migration = None

        self.differential_spectrum = None
        self.differential_spectrum_err = None

        self.energy_function = None

        self._effective_area_recently_set = False

        self.stats = {}

    ##############################################################
    # Define functions to set variables
    ##############################################################

    def set_energy_binning(self, ebins):
        self.energy_binning = ebins

    def set_zenith_binning(self, zdbins):
        self.zenith_binning = zdbins

    def set_theta_square(self, theta_square):
        self.theta_square = theta_square

    def set_correction_factors(self, true_or_false=True):
        self.use_correction_factors = true_or_false

    def set_alpha(self, alpha):
        self.alpha = alpha

    ##############################################################
    # Define functions to set used data and mc
    ##############################################################

    def set_list_of_ceres_files(self, path):
        self.list_of_ceres_files = path

    def set_ganymed_file_mc(self, path):
        self.ganymed_file_mc = path

    def set_ganymed_file_data(self, path):
        self.ganymed_file_data = path

    def set_run_list_star(self, star_list):
        self.run_list_star = star_list

    ##############################################################
    # Optimise Theta and optimise e-binning
    ##############################################################
    def read_events(self):
        if self.energy_function is None:
            def energy_function(x):

                return ((np.log10(x['MHillas.fSize'])+0.7)-3)/1.22+3.05
        else:
            energy_function = self.energy_function


        select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal',
                         'MTime.fMjd', 'MTime.fTime.fMilliSec', 'MTime.fNanoSec',
                         'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2',
                         'MHillas.fLength', 'MHillas.fWidth']
        data = read_mars(self.ganymed_file_data, leaf_names=select_leaves)
        data["energy"] = energy_function(data)
        return data

    def optimize_theta(self):
        events = self.read_events()

        def overall_sigma(x, data=None):
            source_data = data.loc[data["ThetaSquared.fVal"] < x]
            on_data = len(source_data.loc[source_data["DataType.fVal"] == 1.0])
            off_data = len(source_data.loc[source_data["DataType.fVal"] == 0.0])
            return 100 - li_ma_significance(on_data, off_data)

        result = minimize_scalar(overall_sigma, bounds=[0.01, 0.1], method='Bounded', args=events)
        self.theta_square = result.x
        return result

    def optimize_ebinning(self,
                          sigma_threshold=2.5,
                          min_bin_percentage=0.4,
                          min_counts_per_bin=10,
                          start_from_low_energy=True):

        data = self.read_events()

        source_data = data.loc[data["ThetaSquared.fVal"] < self.theta_square]
        on_data = source_data.loc[source_data["DataType.fVal"] == 1.0]
        off_data = source_data.loc[source_data["DataType.fVal"] == 0.0]

        on_data = on_data.copy()
        off_data = off_data.copy()

        on_data.sort_values("energy", ascending=start_from_low_energy, inplace=True)
        off_data.sort_values("energy", ascending=start_from_low_energy, inplace=True)

        sigma_per_bin = sigma_threshold
        sigma_list = []

        def calc_and_append():
            low_index = bin_edges[-1]
            high_index = i
            n_on = len(on_data.iloc[low_index:high_index])
            n_off = len(off_data.loc[(off_data.energy >= on_data.iloc[low_index].energy) & (
                off_data.energy <= on_data.iloc[high_index].energy)])
            sigma_li_ma = li_ma_significance(n_on, n_off)
            nexcess = n_on * self.alpha * n_off
            e_high = on_data.iloc[high_index].energy
            e_low = on_data.iloc[low_index].energy
            size = np.abs((e_high - e_low) / e_low)
            if (((sigma_li_ma >= sigma_per_bin) & (size > min_bin_percentage) &
                 (nexcess > min_counts_per_bin)) | (i == length - 1)):
                bin_edges.append(high_index)
                energy = int(on_data.iloc[high_index - 1].energy) + 1
                if energy != bin_edges_energy[-1]:
                    bin_edges_energy.append(energy)
                    sigma_list.append(sigma_li_ma)

        if start_from_low_energy:
            min_energy = int(on_data.iloc[0].energy * 1.5) + 1
            bin_edges = [0, on_data.energy.searchsorted(min_energy)[0]]
            bin_edges_energy = [int(on_data.iloc[0].energy) + 1, min_energy]

            length = len(on_data)
            if length > 1:
                for i in tqdm(range(length)):
                    calc_and_append()
            else:
                print("Only {} events, skipping bin optimization".format(length))
                bin_edges_energy = self.energy_binning
        else:
            bin_edges = [0]
            bin_edges_energy = [int(on_data.iloc[0].energy) + 1]

            length = len(on_data)
            if length > 1:
                for i in tqdm(range(length)):
                    calc_and_append()
            else:
                print("Only {} events, skipping bin optimization".format(length))
                bin_edges_energy = self.energy_binning

        print(bin_edges_energy)
        self.energy_binning = np.array(np.sort(bin_edges_energy), dtype=np.float)

    ##############################################################
    # Define functions to read data and calculate spectra
    ##############################################################

    def calc_ontime(self, data=None, n_chunks=8, use_multiprocessing=True, force_calc=False):
        if self.on_time_per_zd is not None:
            if (not force_calc) and np.array_equal(self.zenith_binning, self.zenith_binning_on_time):
                print("Skipping on-time calculation, because zenith binning has not changed.")
                return self.on_time_per_zd

        if data is not None:
            self.run_list_star = data
        if self.run_list_star is None:
            print('No list of star-files given, please provide one')
        self.on_time_per_zd = calc_on_time(self.run_list_star,
                                           self.zenith_binning,
                                           n_chunks=n_chunks,
                                           use_multiprocessing=use_multiprocessing)
        # Save binning to check if the binning has changed and the on time calculation skipped
        self.zenith_binning_on_time = self.zenith_binning
        self.total_on_time = np.sum(self.on_time_per_zd)
        print("On Time per ZD:", self.on_time_per_zd)
        return self.on_time_per_zd

    def calc_on_off_histo(self, ganymed_file=None, cut=None, use_multiprocessing=True):
        select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                         'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2',
                         'MHillas.fWidth', 'MHillasSrc.fDist', 'MHillasExt.fM3Long',
                         'MHillasExt.fSlopeLong',
                         'MHillas.fLength',
                         'MHillasExt.fSlopeSpreadWeighted',
                         'MHillasExt.fTimeSpreadWeighted',
                         'MHillasSrc.fCosDeltaAlpha']
        if ganymed_file:
            self.ganymed_file_data = ganymed_file

        if self.ganymed_file_data is None:
            leafs = select_leaves.copy()
            leafs.remove('DataType.fVal')
            leafs.remove('FileId.fVal')
            leafs.remove('ThetaSquared.fVal')
            leafs.remove('MPointingPos.fZd')

            histos = histos_from_list_of_mars_files(self.run_list_star,
                                                    leafs,
                                                    self.zenith_binning,
                                                    self.energy_binning,
                                                    self.theta_square,
                                                    efunc=self.energy_function,
                                                    cut_function=cut,
                                                    use_multiprocessing=use_multiprocessing)
        else:
            data_cut = read_mars(self.ganymed_file_data, leaf_names=select_leaves)
            histos = calc_onoffhisto(data_cut,
                                     self.zenith_binning,
                                     self.energy_binning,
                                     self.theta_square,
                                     energy_function=self.energy_function,
                                     cut=cut)

        # Save Theta-Sqare histograms
        self.theta_square_binning = histos[1][0][1]
        self.on_theta_square_histo = histos[1][0][0]
        self.off_theta_square_histo = histos[1][1][0]

        # Zenith, Energy histograms
        self.on_histo_zenith = histos[0][0]
        self.off_histo_zenith = histos[0][1]

        self.excess_histo = self.on_histo_zenith - self.alpha * self.off_histo_zenith
        self.excess_histo_err = np.sqrt(self.on_histo_zenith + self.alpha**2 * self.off_histo_zenith)

        # Energy histograms

        self.on_histo = np.sum(self.on_histo_zenith, axis=0)
        self.off_histo = np.sum(self.off_histo_zenith, axis=0)

        self.excess_histo = self.on_histo - self.alpha * self.off_histo
        self.excess_histo_err = np.sqrt(self.on_histo + self.alpha**2 * self.off_histo)

        self.significance_histo = li_ma_significance(self.on_histo, self.off_histo, self.alpha)

        # Calculate overall statistics

        self.n_on_events = np.sum(self.on_histo_zenith)
        self.n_off_events = np.sum(self.off_histo_zenith)

        self.n_excess_events = self.n_on_events - self.alpha * self.n_off_events
        self.n_excess_events_err = np.sqrt(self.n_on_events + self.alpha**2 * self.n_off_events)

        self.overall_significance = li_ma_significance(self.n_on_events, self.n_off_events, self.alpha)

    def calc_effective_area(self, analysed_ceres_ganymed=None, ceres_list=None, slope_goal=None, cut=None):
        if not ceres_list:
            ceres_list = self.list_of_ceres_files
        if not analysed_ceres_ganymed:
            analysed_ceres_ganymed = self.ganymed_file_mc

        areas = calc_a_eff_parallel_hd5(self.energy_binning,
                                        self.zenith_binning,
                                        self.use_correction_factors,
                                        self.theta_square,
                                        path=analysed_ceres_ganymed,
                                        list_of_hdf_ceres_files=ceres_list,
                                        energy_function=self.energy_function,
                                        slope_goal=slope_goal,
                                        impact_max=54000.0,
                                        cut=cut)

        self.effective_area, self.effective_area_err, self.energy_migration = areas
        return self.effective_area, self.effective_area_err

    def set_effective_area(self, area_and_mig):
        """ Function to set the effective area. area_and_mig must be a tuple or list of the following order:
            effective area, error of effective area, energy_migration matrice and binning [ebins, zdbins]."""
        self.effective_area, self.effective_area_err, self.energy_migration = area_and_mig
        self._effective_area_recently_set = True

    def calc_differential_spectrum(self,
                                   use_multiprocessing=True,
                                   efunc=None,
                                   slope_goal=None,
                                   force_calc=False,
                                   cut=None,
                                   filename=None):
        if efunc is None:
            if self.energy_function is not None:
                efunc = self.energy_function
        else:
            self.energy_function = efunc

        if (self.on_time_per_zd is None) or force_calc:
            self.calc_ontime(use_multiprocessing=use_multiprocessing, force_calc=force_calc)

        if (self.on_histo is None) or force_calc:
            self.calc_on_off_histo(cut=cut, use_multiprocessing=use_multiprocessing)

        if (self.effective_area is None) or force_calc:
            if not self._effective_area_recently_set:
                self.calc_effective_area(slope_goal=slope_goal, cut=cut)

        bin_centers = np.power(10, (np.log10(self.energy_binning[1:]) + np.log10(self.energy_binning[:-1])) / 2)
        bin_width = self.energy_binning[1:] - self.energy_binning[:-1]

        bin_error = np.array([bin_centers - self.energy_binning[:-1], self.energy_binning[1:] - bin_centers])

        self.scaled_effective_area = (self.effective_area * self.on_time_per_zd[:, np.newaxis]) / self.total_on_time
        self.scaled_effective_area_err = np.divide(self.effective_area_err * self.on_time_per_zd[:, np.newaxis],
                                                   self.total_on_time)

        flux = np.divide(self.excess_histo, np.sum(self.scaled_effective_area, axis=0))
        flux = np.divide(flux, self.total_on_time)
        exc_err = np.sqrt(self.on_histo + (1 / 25) * self.off_histo)
        scaled_a_eff=np.sum(self.scaled_effective_area, axis=0)
        a_eff_err = np.sqrt(np.sum(self.scaled_effective_area_err**2))
        flux_err = flux * np.sqrt(np.ma.divide(exc_err, self.excess_histo)**2 +
                                  np.ma.divide(a_eff_err, np.sum(self.scaled_effective_area, axis=0))**2)

        flux_de = np.divide(flux, np.divide(bin_width, 1000))
        flux_de_err = np.divide(flux_err, np.divide(bin_width, 1000))  # / (flux_de * np.log(10))
        if filename is not None:
            np.savetxt(filename+'.txt', (np.vstack((bin_centers,flux,bin_error,flux_err,flux_err)).T),header='bin_centers flux_de, bin_error (low&high), flux_de_error_log10 (low&high)')
            np.savetxt(filename+'flux_de_err.txt', (np.vstack((bin_centers,flux_de,bin_error,flux_de_err,flux_de_err)).T),header='bin_centers flux_de, bin_error (low&high), flux_de_error_log10 (low&high)')
            np.savetxt(filename+'flux_de_err_andmore.txt', (np.vstack((bin_centers,flux_de,bin_error,flux_de_err,flux_de_err,self.excess_histo,exc_err)).T),header='bin_centers flux_de, bin_error (low&high), flux_de_error_log10 (low&high),ExcessHisto, exc_err')     
            np.savetxt(filename+'Scaled_EffectiveArea.txt', (np.vstack((bin_centers,scaled_a_eff)).T),header='bin_centers scaledEffectiveError') 
            np.savetxt(filename+'EffectiveArea.txt', (np.vstack((bin_centers,np.sum(self.effective_area, axis=0))).T),header='bin_centers EffectiveArea')
            np.savetxt(filename+'error.txt',self.scaled_effective_area_err)
            np.savetxt(filename+'errorSum.txt',np.sum(self.scaled_effective_area_err))
        flux_de_err_log10 = symmetric_log10_errors(flux_de, flux_de_err)
        if filename is not None:
            np.savetxt(filename+'LogErrors.txt', (np.vstack((bin_centers,flux_de,bin_error,flux_de_err_log10)).T),header='bin_centers flux_de, bin_error (low&high), flux_de_error_log10 (low&high)') 
            np.savetxt(filename+'LogErrors_NumberOfPoints.txt', (np.vstack((bin_centers,flux_de,bin_error,flux_de_err_log10,self.excess_histo)).T),header='bin_centers flux_de, bin_error (low&high), flux_de_error_log10 (low&high), Number of Points per Energy Bin')

        self.differential_spectrum = flux_de
        self.differential_spectrum_err = flux_de_err_log10

        self.energy_center = bin_centers
        self.energy_error = bin_error
   
        return flux_de, flux_de_err_log10, bin_centers, bin_error

    ##########################################################
    # Wrapper methods for plotting
    ##########################################################

    def fill_stats(self):
        self.stats["n_on"] = self.n_on_events
        self.stats["n_off"] = self.alpha * self.n_off_events
        self.stats["n_excess"] = self.n_excess_events
        self.stats["on_time_hours"] = self.total_on_time / (60 * 60)
        self.stats["significance"] = self.overall_significance

    def info(self):
        self.fill_stats()
        print(self.stats)

    def plot_flux(self, **kwargs):
        if self.differential_spectrum is None:
            print("No differential spectrum data, please run Spectrum.calc_differential_spectrum()")
            return
        axes = plot_spectrum(self.energy_center,
                             self.energy_error,
                             self.differential_spectrum,
                             self.differential_spectrum_err,
                             self.significance_histo,
                             **kwargs)
        return axes

    def plot_thetasq(self, ax=None, **kwargs):
        if self.on_theta_square_histo is None:
            print("No theta square histo, please run Spectrum.calc_differential_spectrum()")
            return
        self.fill_stats()
        return plot_theta(self.theta_square_binning,
                          self.on_theta_square_histo,
                          self.off_theta_square_histo,
                          self.theta_square,
                          self.stats,
                          ax=ax,
                          **kwargs)

    ##############################################################
    # Define functions to dump and load variables as json
    ##############################################################

    def save(self, filename):
        save_variables_to_json(self, filename)

    def load(self, filename):
        load_variables_from_json(self, filename)
