from collections.abc import Sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from subprocess import run
from os import remove
import multiprocessing as mp
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from blockspec.spec import Spectrum
from blockspec.spec.plotting import plot_spectrum
from blockspec.spec.spectrum_class import load_variables_from_json, save_variables_to_json

import matplotlib.pyplot as plt


def confidence(popt, pcov, conf_low=16, conf_up=84):
    sample = np.random.multivariate_normal(popt, pcov, 10000)
    x3 = np.logspace(np.log10(0.750), np.log10(50), 100)
    linepoints = np.power(10, line(np.log10(x3), popt[0], popt[1]))
    y3 = np.power(10, line(np.log10(x3)[:, np.newaxis], sample[:, 0], sample[:, 1]))
    low = np.percentile(y3, conf_low, axis=1)
    up = np.percentile(y3, conf_up, axis=1)

    x3 *= 1000
    return x3, low, up


def line(x, a, b):
    return a - b * x


def notlog(x, a, b):
    return np.power(10, line(np.log10(x), a, b))


def blocks_from_json(filename):
    blocks = BlockAnalysis()
    blocks.load(filename)
    return blocks


class BlockAnalysis(Sequence):

    list_of_variables = ["ceres_list",
                         "ganymed_mc",
                         "basepath",
                         "ganymed_path",
                         "mars_directory",
                         "spectrum_path",
                         "source_name",
                         "source_ra",
                         "source_deg",
                         "fitvalues",
                         "fitvalues2",
                         "tfitvalues",
                         "tfitvalues2"]

    def __init__(self,
                 ceres_list=["/home/michi/ceres.h5"],
                 ganymed_mc="/media/michi/523E69793E69574F/daten/gamma/hzd_gammasall-analysis.root",
                 basepath='/media/michi/523E69793E69574F/daten/Crab/',
                 ganymed_path="/media/michi/523E69793E69574F/gamma/ganymed.C",
                 mars_directory="/home/michi/Mars/",
                 spec_identifier="forest2_",
                 source_name=None,
                 ):

        super().__init__()
        self.ceres_list = ceres_list
        print("List of ceres lists:", self.ceres_list)

        self.ganymed_mc = ganymed_mc
        print("Ganymed File of analyzed MC:", self.ganymed_mc)

        self.basepath = basepath
        print("Basepath of block analysis:", self.basepath)

        self.ganymed_path = ganymed_path
        print("Ganymed File of analyzed data:", self.ganymed_path)

        self.mars_directory = mars_directory
        print("Mars Directory:", self.mars_directory)

        self.spectrum_path = basepath + spec_identifier
        print("Saving spectra to prefix", self.spectrum_path)

        self.source_name = source_name

        self.source_ra = None
        self.source_deg = None

        if source_name:
            try:
                source = SkyCoord.from_name(self.source_name)
                self.source_ra = source.ra.hour
                self.source_deg = source.dec.deg
            except NameResolveError:
                print("Temporary failure in name resolution, initializing source coordinates with None")

        print("Source:", self.source_name)
        print("    Ra:", self.source_ra)
        print("    Dec:", self.source_deg)

        self.mapping = None
        self.load_mapping()

        self.spectra = []
        self.spectral_data = []

        self.fitvalues = []
        self.fitvalues2 = []
        self.tfitvalues = []
        self.tfitvalues2 = []

        if self.mapping is not None:
            try:
                print("Trying to load spectral data.")
                self.load_spectral_data()
                print("Loading spectral data succeeded.")
            except FileNotFoundError:
                print("Loading spectral data failed. File not found.")

    def __getitem__(self, i):
        return self.mapping.iloc[i], self.spectra[i]

    def __len__(self):
        return len(self.spectra)

    def save(self, filename):
        save_variables_to_json(self, filename)
        self.mapping.to_json(self.basepath + "blocks/mapping.json", double_precision=15)

    def load(self, filename):
        load_variables_from_json(self, filename)
        self.load_mapping()
        self.load_spectral_data()

    def load_mapping(self):
        try:
            self.mapping = pd.read_json(self.basepath + "blocks/mapping.json", precise_float=True)
            print("Mapping succesfully read.")

        except ValueError:
            print("Mapping file does not yet exist.")

        if self.mapping is not None:
            self.calc_mapping_columns()

    def run_spectra(self,
                    theta_sq=0.04,
                    efunc=None,
                    correction_factors=False,
                    optimize_theta=False,
                    start=None,
                    stop=None,
                    force=False):

        if "has_data" in self.mapping.columns:
            has_data = self.mapping["has_data"].values
        else:
            has_data = np.full(len(self.mapping), False)

        for element in self.mapping.iloc[start:stop].itertuples():
            block_number = element[0]
            filepath = element[1]
            star_files = [filepath]
            star_list = []
            for entry in star_files:
                star_list += list(open(entry, "r"))

            ganymed_result = self.basepath + str(block_number) + "-analysis.root"

            path = self.spectrum_path + str(block_number) + ".json"

            spectrum = Spectrum(run_list_star=star_list,
                                ganymed_file_data=ganymed_result,
                                list_of_ceres_files=self.ceres_list,
                                ganymed_file_mc=self.ganymed_mc,
                                theta_sq=theta_sq)

            try:
                spectrum.load(path)
                calc = False
            except FileNotFoundError:
                calc = True

            if force or calc:
                spectrum.set_correction_factors(correction_factors)

                print("Trying to calculate spectrum for block", block_number)

                try:
                    if optimize_theta:
                        spectrum.optimize_theta()

                    spectrum.optimize_ebinning(sigma_threshold=1.3, min_counts_per_bin=10)
                    # spectrum.set_energy_binning(np.logspace(np.log10(200), np.log10(50000), 10))

                    spectrum.calc_differential_spectrum(efunc=efunc, force_calc=force)

                    spectrum.save(path)
                    self.spectra.append(spectrum)
                    has_data[block_number] = True

                except AttributeError as err:
                    print(err)
                    print("Please check if there is any star file in block", block_number)
                    has_data[block_number] = False

        self.mapping["has_data"] = has_data

    def run_ganymed(self, index, path):
        outfile = self.basepath + str(index)
        execute = "root -b -q '" + self.ganymed_path + "(" + '"' + path + '",' + '"' + outfile + '",' + str(
            self.source_ra) + ", " + str(self.source_deg) + ")'"
        run(execute, cwd=self.mars_directory, shell=True)
        remove(outfile + "-summary.root")

    def execute_ganymeds(self, multiprocessing=True, start=None, stop=None):
        if multiprocessing:
            pool = mp.Pool()
            result = [pool.apply_async(self.run_ganymed, args=(element[0], element[1]))
                      for element in self.mapping.iloc[start:stop].itertuples()]
            pool.close()
            pool.join()
        else:
            for element in tqdm(self.mapping.iloc[start:stop].itertuples()):
                self.run_ganymed(element[0], element[1])

    def calc_mapping_columns(self):
        mapping = self.mapping.copy()
        mapping.sort_index(inplace=True)
        start = pd.to_datetime(mapping.start, unit="ms")
        stop = pd.to_datetime(mapping.stop, unit="ms")
        mapping["start"] = start
        mapping["stop"] = stop
        mapping["time_err"] = (stop - start)/2
        mapping["time"] = mapping.start + mapping.time_err
        self.mapping = mapping.copy()

    def load_spectral_data(self, start=None, stop=None):
        self.spectral_data = []
        self.spectra = []
        if start or stop:
            selection = self.mapping.iloc[start:stop].index.values
        elif "has_data" in self.mapping.columns:
            selection = self.mapping[self.mapping.has_data].index.values
        else:
            selection = self.mapping.index.values

        for element in self.mapping.iloc[selection].itertuples():
            block_number = element[0]
            spect = Spectrum()
            path = self.spectrum_path + str(block_number) + ".json"
            print("Trying to load", path)
            spect.load(path)
            self.spectra.append(spect)
            with open(self.spectrum_path + str(block_number) + ".json") as infile:
                data = json.load(infile)
            self.spectral_data.append(data)

        for i in range(len(self.spectral_data)):
            for variable_name in data:
                containing = self.spectral_data[i][variable_name]
                if isinstance(containing, list):
                    containing = np.array(containing)
                    if None in containing:  # allows to load masked numpy arrays
                        containing = np.ma.masked_invalid(containing.astype(np.float))
                self.spectral_data[i][variable_name] = containing

    def run_fits(self, start=None, stop=None):
        from scipy.optimize import curve_fit
        from scipy.integrate import quad
        from scipy import integrate

        if len(self.spectral_data) == 0:
            self.load_spectral_data(start, stop)

        def exp_pow(x, a, b, c):
            return a * np.power(x / 1000, -b - c * np.log10(x / 1000))

        fitvalues = []
        for block_number in range(0, len(self.spectral_data)):
            selection = self.spectral_data[block_number]["energy_center"] > 700
            x = self.spectral_data[block_number]["energy_center"][selection][1:-2]
            if len(x) > 2:
                y = self.spectral_data[block_number]["differential_spectrum"][selection][1:-2]
                popt, pcov = curve_fit(exp_pow, xdata=x, ydata=y, p0=[3 * 10 ** (-11), 2.6, 0.24])
                pcov = np.sqrt(np.diag(pcov))
                fitvalues.append([block_number, popt[0], popt[1], popt[2], pcov[0], pcov[1], pcov[2]])
            else:
                fitvalues.append([block_number, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        fitvalues = np.array(fitvalues)


        def line(x, a, b):
            return a - b * x

        def notlog(x, a, b):
            return np.power(10, line(np.log10(x), a, b))

        high_bin = -1
        pcovs = []
        fitvalues2 = []
        for block_number in tqdm(range(0, len(self.spectral_data))):
            # print(block_number)
            selection = (self.spectral_data[block_number]["energy_center"] > 1000) & (
                self.spectral_data[block_number]["significance_histo"] > 1.3)
            x = np.log10(self.spectral_data[block_number]["energy_center"][selection][:high_bin] / 1000)
            # print(x)
            if len(x) > 2:
                # print(spectral_data[block_number]["differential_spectrum"][selection][:high_bin])
                y = np.log10(np.abs(self.spectral_data[block_number]["differential_spectrum"][selection][:high_bin]))
                # print(y)
                popt, pcov = curve_fit(line, xdata=x, ydata=y,
                                       # sigma=np.abs(spectral_data[block_number]["differential_spectrum_err"][0][selection][:high_bin]),
                                       p0=[10 ** (-2), 2.0])
                pcovs.append([popt, pcov])
                sample = np.random.multivariate_normal(popt, pcov, 10000)
                pcov = np.sqrt(np.diag(pcov))

                x3 = np.logspace(np.log10(1), np.log10(50), 200)
                linepoints = np.power(10, line(np.log10(x3), popt[0], popt[1]))
                y3 = np.power(10, line(np.log10(x3)[:, np.newaxis], sample[:, 0], sample[:, 1]))
                low = np.percentile(y3, 16, axis=1)
                up = np.percentile(y3, 84, axis=1)
                low90 = np.percentile(y3, 5, axis=1)
                up90 = np.percentile(y3, 95, axis=1)

                y2 = self.spectral_data[block_number]["differential_spectrum"][selection][:high_bin]
                x2 = self.spectral_data[block_number]["energy_center"][selection][:high_bin] / 1000
                y2_el = y2 - self.spectral_data[block_number]["differential_spectrum_err"][0][selection][:high_bin]
                y2_eh = y2 + self.spectral_data[block_number]["differential_spectrum_err"][1][selection][:high_bin]

                fitvalues2.append(
                    [block_number, popt[0], popt[1], np.trapz(y=y2, x=x2), pcov[0], pcov[1], np.trapz(y=y2_el, x=x2),
                     np.trapz(y=y2_eh, x=x2), np.power(10, line(np.log10(1), *popt)),
                     quad(notlog, 1, np.inf, args=(popt[0], popt[1]))[0], integrate.simps(low, x3),
                     integrate.simps(linepoints, x3), integrate.simps(up, x3)])
            else:
                fitvalues2.append(
                    [block_number, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan])
                pcovs.append(None)
        fitvalues2 = np.array(fitvalues2)

        self.fitvalues = fitvalues
        self.fitvalues2 = fitvalues2

        self.pcovs = pcovs

        self.tfitvalues = fitvalues.T
        self.tfitvalues2 = fitvalues2.T

        # if there is - for some reason - no spectral data for the last blocks, fill the missing fitvalues with nan:
        if len(self.mapping) > len(self.tfitvalues2):
            fillnans = np.full((self.tfitvalues2.shape[0], len(self.mapping) - self.tfitvalues.shape[1]),
                               np.nan)

            self.tfitvalues2 = np.concatenate((self.tfitvalues2, fillnans), axis=1)
            self.tfitvalues = np.concatenate((self.tfitvalues, fillnans[:self.tfitvalues.shape[0],:]), axis=1)

        self.mapping["photon_index"] = self.tfitvalues2[2]
        self.mapping["photon_index_err"] = self.tfitvalues2[5]

        self.mapping["flux1"] = self.tfitvalues2[3]
        self.mapping["flux1_err_lower"] = self.tfitvalues2[6]
        self.mapping["flux1_err_upper"] = self.tfitvalues2[7]

        self.mapping["flux2"] = self.tfitvalues[1]
        self.mapping["flux2_err"] = self.tfitvalues[3]
        self.mapping["flux3"] = self.tfitvalues2[8]
        self.mapping["flux4"] = self.tfitvalues2[9]
        self.mapping["flux5"] = self.tfitvalues2[11]
        self.mapping["flux5up"] = self.tfitvalues2[12] - self.mapping.flux5
        self.mapping["flux5low"] = self.mapping.flux5 - self.tfitvalues2[10]

    def plot_fluxes(self, name=None):

        if not self.check_for_data():
            return 0

        if len(self.fitvalues) < 1:
            print("No fitvalues present, please run BlockAnalysis.run_fits()")
            return 0

        if name is not None:
            to_pdf = True
            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(name)
        else:
            to_pdf = False

        for block_number in range(len(self.spectral_data)):

            self.spectra[block_number].plot_flux(crab_do=True,
                            label=str(block_number) + ": " + self.mapping.time[block_number].strftime("%Y-%m-%d %H:%M"))

            x = np.logspace(np.log10(0.750), np.log10(50), 100)
            fit_select = self.fitvalues[:, 0] == block_number
            if fit_select.any():
                if not np.isnan(self.mapping["photon_index"][block_number]):
                    # plt.plot(x, exp_pow(x, *fitvalues[fit_select][0,1:3]), label="Exponential Fit")
                    # plt.plot(x, exp_pow(x, fitvalues[fit_select][0][1], fitvalues[fit_select][0][2]+0.5), label="Fit +0.5")
                    # plt.plot(x, exp_pow(x, fitvalues[fit_select][0][1], fitvalues[fit_select][0][2]-0.5), label="Fit -0.5")
                    plt.plot(x * 1000, np.power(10, line(np.log10(x), self.fitvalues2[fit_select][0][1],
                                                         self.fitvalues2[fit_select][0][2])), label="Linear Fit")
                    text = "Index: {0:1.2f} $\pm$ {1:1.2f} \nLog10(Flux) at 1 TeV:\n{2:.2f} $\pm$ {3:2.2f}".format(
                        self.tfitvalues2[2][block_number], self.tfitvalues2[5][block_number], self.tfitvalues2[1][block_number],
                        self.tfitvalues2[4][block_number])
                    plt.fill_between(*confidence(*self.pcovs[block_number]), alpha=0.4, label="68% containment")
                    plt.plot([], [], ' ', label=text)
                plt.legend()
                plt.tight_layout()
                # plt.savefig("/home/michi/mrk501/"+str(block_number)+"-flux.pdf", format="pdf")
                if to_pdf:
                    pp.savefig()

        if to_pdf:
            pp.close()
        plt.show()

    def check_for_data(self):
        if len(self.spectra) > 0:
            "Spectra present."
            spectra  = True
        else:
            print("No spectral data present, please execute the spectra calculation via BlockAnalysis.run_spectra().")
            spectra = False
        if len(self.spectral_data) > 0:
            "Spectral fit data present"
            spectral_data = True
        else:
            print("Fit data missing, please execute BlockAnalysis.run_fits()")
            spectral_data = False

        if (spectra & spectral_data):
            return True
        else:
            return False

    def plot_thetasq(self, name=None):

        if not self.check_for_data():
            return 0

        if name is not None:
            to_pdf = True
            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(name)
        else:
            to_pdf = False

        for block_number in range(len(self.spectral_data)):
            #plt.figure()
            self.spectra[block_number].plot_thetasq()
            plt.title(str(block_number) + ": " + self.mapping.time[block_number].strftime("%Y-%m-%d %H:%M"))
            if to_pdf:
                pp.savefig()

        if to_pdf:
            pp.close()
        plt.show()

    def plot_all_spectra(self):
        fig = plt.figure()
        ax_sig = plt.subplot2grid((8, 1), (6, 0), rowspan=2)  # inspired from pyfact
        ax_flux = plt.subplot2grid((8, 1), (0, 0), rowspan=6, sharex=ax_sig)
        for block_number in range(len(self.spectral_data)):
            plot_spectrum(self.spectral_data[block_number]["energy_center"], self.spectral_data[block_number]["energy_error"],
                          self.spectral_data[block_number]["differential_spectrum"],
                          self.spectral_data[block_number]["differential_spectrum_err"],
                          self.spectral_data[block_number]["significance_histo"], ax_flux=ax_flux, ax_sig=ax_sig,
                          label=str(block_number), alpha=0.3)
        plt.show()

    def plot_index_vs_flux(self):
        plt.errorbar(x=self.mapping.flux5, xerr=[self.mapping.flux5low, self.mapping.flux5up], y=self.mapping.photon_index,
                     yerr=self.mapping.photon_index_err, fmt=".", label="Crab")
        plt.xscale("log")
        plt.xlabel("Flux [$\mathrm{cm^{-2}s^{-1}}$]")
        plt.ylabel("Photon Index")
        plt.grid()
        plt.show()