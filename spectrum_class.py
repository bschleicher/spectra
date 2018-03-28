import numpy as np
from read_mars import read_mars
from fact.analysis.statistics import li_ma_significance

from calc_a_eff_parallel import calc_a_eff_parallel_hd5
from read_data import histos_from_list_of_mars_files, calc_onoffhisto
from on_time_parallel import calc_on_time
from spectrum_eff_eff_area import symmetric_log10_errors
from plotting import plot_spectrum, plot_theta


class Spectrum:
    """ Class containing FACT spectra and additional information"""

    def __init__(self,
                 run_list_star=None,
                 theta_sq=0.085,
                 correction_factors=False,
                 ebins=None,
                 elabels=None,
                 zdbins=None,
                 zdlabels=None,
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

        if elabels:
            self.energy_labels = elabels
        else:
            self.energy_labels = range(len(self.energy_binning) - 1)

        if zdlabels:
            self.zenith_labels = zdlabels
        else:
            self.zenith_labels = range(len(self.zenith_binning) - 1)
        if ganymed_file_data:
            self.ganymed_file_data = ganymed_file_data

        # Declare Placeholder variables
        self.on_time_per_zd = None
        self.total_on_time = None

        self.on_histo_zenith = None
        self.off_histo_zenith = None
        self.on_histo = None
        self.off_histo = None
        self.excess_histo = None
        self.excess_histo_err = None
        self.n_on_events = None
        self.n_off_events = None
        self.n_excess_events = None
        self.n_excess_events_err = None
        self.overall_significance = None

        self.on_theta_square_histo = None
        self.off_theta_square_histo = None

        self.effective_area = None
        self.scaled_effective_area = None

        self.differential_spectrum = None
        self.differential_spectrum_err = None

        self.stats = {}

    ##############################################################
    # Define functions to set variables
    ##############################################################

    def set_energy_binning(self, ebins, elabels=None):
        self.energy_binning = ebins
        if elabels:
            self.energy_labels = elabels
        else:
            self.energy_labels = range(len(ebins)-1)

    def set_zenith_binning(self, zdbins, zdlabels=None):
        self.zenith_binning = zdbins
        if zdlabels:
            self.zenith_labels = zdlabels
        else:
            self.zenith_labels = range(len(zdbins) - 1)

    def set_theta_square(self, theta_square):
        self.theta_square = theta_square

    def use_correction_factors(self, true_or_false=True):
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
    # Define functions to read data and calculate spectra
    ##############################################################

    def calc_ontime(self, data=None, n_chunks=8):
        if data:
            self.run_list_star = data
        if not self.run_list_star:
            print('No list of star-files given, please provide one')
        self.on_time_per_zd = calc_on_time(self.run_list_star, self.zenith_binning, self.zenith_labels, n_chunks)

        self.total_on_time = np.sum(self.on_time_per_zd)

    def calc_on_off_histo(self, ganymed_file=None):
        select_leaves = ['DataType.fVal', 'MPointingPos.fZd', 'FileId.fVal', 'MTime.fMjd', 'MTime.fTime.fMilliSec',
                         'MTime.fNanoSec', 'MHillas.fSize', 'ThetaSquared.fVal', 'MNewImagePar.fLeakage2']
        if ganymed_file:
            self.ganymed_file_data = ganymed_file

        if not self.ganymed_file_data:
            histos = histos_from_list_of_mars_files(self.run_list_star,
                                                    select_leaves,
                                                    self.zenith_binning,
                                                    self.zenith_labels,
                                                    self.energy_binning,
                                                    self.energy_labels,
                                                    self.theta_square)
        else:
            data_cut = read_mars(self.ganymed_file_data, leaf_names=select_leaves)
            histos = calc_onoffhisto(data_cut, self.zenith_binning, self.zenith_labels, self.energy_binning,
                                     self.energy_labels, self.theta_square)

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

        # Save Theta-Sqare histograms

        self.on_theta_square_histo = histos[1][0]
        self.off_theta_square_histo = histos[1][1]

    def calc_effective_area(self, analysed_ceres_ganymed=None, ceres_list=None):
        if not ceres_list:
            ceres_list = self.list_of_ceres_files
        if not analysed_ceres_ganymed:
            analysed_ceres_ganymed = self.ganymed_file_mc

        self.effective_area = calc_a_eff_parallel_hd5(self.energy_binning,
                                                      self.zenith_binning,
                                                      self.use_correction_factors,
                                                      self.theta_square,
                                                      path=analysed_ceres_ganymed,
                                                      list_of_hdf_ceres_files=ceres_list)
        return self.effective_area

    def calc_differential_spectrum(self):

        if not self.on_time_per_zd:
            self.calc_ontime()

        if not self.on_histo:
            self.calc_on_off_histo()

        if not self.effective_area:
            self.calc_effective_area()

        bin_centers = np.power(10, (np.log10(self.energy_binning[1:]) + np.log10(self.energy_binning[:-1])) / 2)
        bin_width = self.energy_binning[1:] - self.energy_binning[:-1]

        bin_error = [bin_centers - self.energy_binning[:-1], self.energy_binning[1:] - bin_centers]

        self.scaled_effective_area = (self.effective_area * self.on_time_per_zd[:, np.newaxis]) / self.total_on_time
        flux = np.divide(self.excess_histo, np.sum(self.scaled_effective_area, axis=0))
        flux = np.divide(flux, self.total_on_time)
        flux_err = np.ma.divide(np.sqrt(self.on_histo + (1 / 25) * self.off_histo),
                                np.sum(self.scaled_effective_area, axis=0)) / self.total_on_time

        flux_de = np.divide(flux, np.divide(bin_width, 1000))
        flux_de_err = np.divide(flux_err, np.divide(bin_width, 1000))  # / (flux_de * np.log(10))
        flux_de_err_log10 = symmetric_log10_errors(flux_de, flux_de_err)

        self.differential_spectrum = flux_de
        self.differential_spectrum_err = flux_de_err_log10

        self.energy_center = bin_centers
        self.energy_error = bin_error

        return flux_de, flux_de_err_log10, bin_centers, bin_error

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
        axes = plot_spectrum(self.energy_center,
                             self.energy_error,
                             self.differential_spectrum,
                             self.differential_spectrum_err,
                             self.significance_histo,
                             **kwargs)
        return axes

    def plot_thetasq(self):
        self.fill_stats()
        return plot_theta(self.on_theta_square_histo, self.off_theta_square_histo, self.theta_square, self.stats)




