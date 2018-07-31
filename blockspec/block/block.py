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
import corner

from blockspec.spec import Spectrum
from blockspec.spec.plotting import plot_spectrum
from blockspec.spec.spectrum_class import load_variables_from_json, save_variables_to_json
from blockspec.block.fitting import fit_ll, fit_points, powerlaw_model, cutoff_powerlaw_model, line_model

import matplotlib.pyplot as plt


def confidence(popt, pcov, conf_low=16, conf_up=84):
    sample = np.random.multivariate_normal(popt, pcov, 10000)
    x3 = np.logspace(np.log10(0.750), np.log10(50), 100)
    # linepoints = np.power(10, line(np.log10(x3), popt[0], popt[1]))
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
                         "tfitvalues2",
                         "ll_dicts",
                         "loglog_dicts"]

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

        self.loglog_dicts = None
        self.ll_dicts = None
        self.fit_results = None

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
        if self.fit_results is not None:
            self.fit_results.to_json(self.basepath + "blocks/fit_results.json", double_precision=15)

    def load(self, filename):
        load_variables_from_json(self, filename)
        self.load_mapping()
        self.load_fits()
        self.load_spectral_data()

    def load_fits(self):
        try:
            self.fit_results = pd.read_json(self.basepath + "blocks/fit_results.json", precise_float=True)
            print("Fit results succesfully read.")

        except ValueError:
            print("File with fit results does not yet exist.")

        if self.mapping is not None:
            self.calc_mapping_columns()

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
                    optimize_ebinning=True,
                    start=None,
                    stop=None,
                    force=False,
                    **kwargs):

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
                                theta_sq=theta_sq,
                                **kwargs)

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

                    if optimize_ebinning:
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
        return True

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

    def _fit(self, fit_function, verbose=False, **kwargs):
        block_numbers = []
        paramvalues = []
        dicts = []
        for i, spect in enumerate(tqdm(self.spectra)):
            block_number = self.mapping[self.mapping["has_data"]].index.values[i]
            block_numbers.append(block_number)
            if verbose:
                print("################################################################")
                print("block number", str(block_number))
                print("################################")
            result_dict = fit_function(spect, **kwargs)
            dicts.append(result_dict)
            paramvalues.append(result_dict["parameters"])

        return block_numbers, paramvalues, dicts

    def _prepare_fit_result_df(self, params, block_numbers, name, names):

        reshaped = params.reshape(params.shape[0], params.shape[1] * params.shape[2])
        lll = ["val", "up", "low"]
        df = pd.DataFrame(reshaped,
                          index=block_numbers,
                          columns=[[name] * len(names) * len(lll), [i for i in names for j in range(len(lll))],
                                   lll * len(names)])
        return df

    def _add_to_fit_results(self, df):
        if self.fit_results is None:
            self.fit_results = df
        else:
            self.fit_results = pd.concat((self.fit_results, df), axis=1)

    def fit_loglog(self, name="linear_fit",
                   verbose=False,
                   model='line',
                   start_values=None,
                   bounds=None,
                   names=None,
                   labels=None,
                   **kwargs):

        if model == 'line':
            fitargs = line_model()
        elif isinstance(model, function):
            if None in (start_values, names, labels):
                raise ValueError('If you provide a function as a model, '
                                 'you also have to provide start_values, names and labels.')
            else:
                fitargs = model, start_values, bounds, names, labels
        else:
            raise ValueError("'model' must either be line or a function")

        block_numbers, paramvalues, dicts = self._fit(fit_points, verbose=verbose, *fitargs, **kwargs)

        self.loglog_dicts = dicts
        setattr(self, name, dicts)

        block_numbers = np.array(block_numbers)
        params = np.array(paramvalues)

        names = self.loglog_dicts[0]["names"]

        df = self._prepare_fit_result_df(params, block_numbers, name, names)
        self._add_to_fit_results(df)

        return df

    def fit_loglike(self,
                    name="ll_powerlaw",
                    verbose=False,
                    model='powerlaw',
                    start_values=None,
                    bounds=None,
                    labels=None,
                    names=None,
                    **kwargs):

        if model == 'powerlaw':
            fit_args = powerlaw_model(bounds=bounds, labels=labels,
                                                                                names=names)

        elif model == 'cutoff_powerlaw':
            fitargs = cutoff_powerlaw_model(bounds=bounds, labels=labels,
                                                                                       names=names)
        elif isinstance(model, function):

            if None in (bounds, names, labels):
                raise ValueError(
                    'If you provide a function as a model, you also have to provide bounds, names and labels.')
            else:
                fitargs = model, start_values, bounds, labels, names
        block_numbers, paramvalues, dicts = self._fit(fit_ll, verbose=verbose, *fitargs, **kwargs)

        self.ll_dicts = dicts
        setattr(self, name, dicts)

        block_numbers = np.array(block_numbers)
        params = np.array(paramvalues)

        names = self.ll_dicts[0]["names"]

        df = self._prepare_fit_result_df(params, block_numbers, name, names)
        self._add_to_fit_results(df)

        return df

    def plot_ll_corner(self, name=None, name_prefix=None, plot_theta_sq=False, plot_flux=False):
        if name is not None:
            to_pdf = True
            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(name)
        else:
            to_pdf = False

        for i in range(len(self.ll_dicts)):

            entry = self.ll_dicts[i]
            k = len(entry["labels"])
            block_number = self.mapping[self.mapping["has_data"]].index.values[i]
            samples = entry["samples"][:, 150:, :].reshape((-1, k))
            plt.figure()
            fig = corner.corner(samples,
                                labels=entry["labels"],
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                title_kwargs={"fontsize": 12})
            if plot_theta_sq or plot_flux:
                fig.set_size_inches(5.5, 5.5)
                fig.set_size_inches(10, 5.5)
                fig.subplots_adjust(right=5.5 / 10)
                if plot_theta_sq:
                    ax = fig.add_axes([6 / 10, 0.15, 4 / 10, 0.4])
                    self.spectra[i].plot_thetasq(ax=ax)

            if name_prefix is not None:
                plt.savefig(self.basepath + name_prefix + str(block_number) + ".png")
            if to_pdf:
                pp.savefig()

        if to_pdf:
            pp.close()
        plt.show()

    def _check_if_str_or_list_group(self, group):
        if isinstance(group, str):
            return [self.fit_results[group]]
        elif isinstance(group, list):
            return [self.fit_results[key] for key in group]
        else:
            raise ValueError("x_group or y_group must be either None when plotting from BlockAnalysis.mapping or a "
                             "name or a list of names of the first level of BlockAnalysis.fit_results")
    def _check_if_str_or_list_key(self, key):
        if isinstance(key, str):
            return [key]
        elif isinstance(key, list):
            return key
        else:
            raise ValueError("x_group or y_group must be either None when plotting from BlockAnalysis.mapping or a "
                             "name or a list of names of the first level of BlockAnalysis.fit_results")

    def _check_if_same_lenth_or_one(self, source, key):
        if len(key) == len(source):
            return key
        elif len(key) == 1:
            return key * len(source)
        else:
            raise ValueError("Shapes of group and key do not match.")


    def plot(self, x_group=None, x_key='time', y_group=None, y_key='flux', **kwargs):
        """ A wrapper function that allows to easily plot parameters of the fit results of the BlockAnalysis"""

        if self.mapping is None:
            raise ValueError("No mapping found, somethings wrong.")

        if self.fit_results is None:
            print("Warning, No fit results!")

        if (x_group is None) and (y_group is None):
            x_source = [self.mapping]
            y_source = [self.mapping]
            x_is_mapping = True
            y_is_mapping = True

        elif x_group is None:
            x_source = [self.mapping.iloc[self.fit_results.index]]
            y_source = self._check_if_str_or_list_group(y_group)
            x_is_mapping = True
            y_is_mapping = False
        elif y_group is None:
            y_source = [self.mapping.iloc[self.fit_results.index]]
            x_source = self._check_if_str_or_list_group(x_group)
            x_is_mapping = False
            y_is_mapping = True
        else:
            x_source = self._check_if_str_or_list_group(x_group)
            y_source = self._check_if_str_or_list_group(y_group)
            x_is_mapping = False
            y_is_mapping = False

        x_key = self._check_if_str_or_list_key(x_key)
        y_key = self._check_if_str_or_list_key(y_key)

        x_key = self._check_if_same_lenth_or_one(x_source, x_key)
        y_key = self._check_if_same_lenth_or_one(y_source, y_key)

        if len(x_source) == len(y_source):
            pass
        elif len(x_source) == 1:
            x_source = x_source * len(y_source)
            x_key = x_key * len(y_source)
        elif len(y_source) == 1:
            y_source = y_source * len(x_source)
            y_key = y_key * len(x_source)
        else:
            raise ValueError("Shapes of x and y do not match! Must be same length or either one must be of length one.")

        for i in range(len(x_source)):
            if x_is_mapping:
                x = x_source[i][x_key[i]].values
                xerr = x_source[i][x_key[i]+"_err"].values
            else:
                x = x_source[i][x_key[i]]["val"].values
                xerr = [x_source[i][x_key[i]]["low"].values, x_source[i][x_key[i]]["up"].values]
            if y_is_mapping:
                y = y_source[i][y_key[i]].values
                yerr = y_source[i][y_key[i]+"_err"].values
            else:
                y = y_source[i][y_key[i]]["val"].values
                yerr = [y_source[i][y_key[i]]["low"].values, y_source[i][y_key[i]]["up"].values]

            plt.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)


    def _plot_flux(self, id, ax_sig=None, ax_flux=None):
        block_number = self.fit_results.index.values[id]
        self.spectra[id].plot_flux(crab_do=True,
                                             label=str(block_number) + ": " + self.mapping.time[block_number].strftime(
                                                 "%Y-%m-%d %H:%M"),
                                             ax_sig=ax_sig,
                                             ax_flux=ax_flux)

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
                    self.tfitvalues2[2][block_number], self.tfitvalues2[5][block_number],
                    self.tfitvalues2[1][block_number], self.tfitvalues2[4][block_number])
                plt.fill_between(*confidence(*self.pcovs[block_number]), alpha=0.4, label="68% containment")
                plt.plot([], [], ' ', label=text)
            plt.legend()



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
            plot_spectrum(self.spectral_data[block_number]["energy_center"],
                          self.spectral_data[block_number]["energy_error"],
                          self.spectral_data[block_number]["differential_spectrum"],
                          self.spectral_data[block_number]["differential_spectrum_err"],
                          self.spectral_data[block_number]["significance_histo"],
                          ax_flux=ax_flux,
                          ax_sig=ax_sig,
                          label=str(block_number),
                          alpha=0.3)
        plt.show()

    def plot_index_vs_flux(self,):
        plt.errorbar(x=self.mapping.flux5,
                     xerr=[self.mapping.flux5low, self.mapping.flux5up],
                     y=self.mapping.photon_index,
                     yerr=self.mapping.photon_index_err,
                     fmt=".",
                     label="Crab")
        plt.xscale("log")
        plt.xlabel("Flux [$\mathrm{cm^{-2}s^{-1}}$]")
        plt.ylabel("Photon Index")
        plt.grid()
        plt.show()