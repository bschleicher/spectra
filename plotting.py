import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spectrum(bin_centers,
                  energy_err,
                  flux,
                  flux_err,
                  significance,
                  hess20140624=False,
                  hess_flare=False,
                  crab_do=False,
                  filename=None,
                  ax_flux=None,
                  ax_sig=None,
                  label="F",
                  **kwargs):

    if (ax_flux is None) and (ax_sig is None):
        fig = plt.figure()

        ax_sig = plt.subplot2grid((8, 1), (6, 0), rowspan=2)  # inspired from pyfact
        ax_flux = plt.subplot2grid((8, 1), (0, 0), rowspan=6, sharex=ax_sig)

    if hess20140624:
        hess_x = np.array((1.70488, 2.1131, 2.51518, 3.02825, 3.65982, 4.43106, 5.37151, 6.50896, 7.87743, 9.52215,
                           11.4901, 13.8626, 16.8379, 20.4584, 24.8479, 30.2065, 36.7507, 44.8404))

        hess_y = np.array((4.15759e-11, 3.30552e-11, 1.7706e-11, 1.28266e-11, 7.57679e-12, 5.65619e-12, 2.85186e-12,
                           1.9475e-12, 1.10729e-12, 4.91077e-13, 3.00283e-13, 8.96491e-14, 4.27756e-14, 1.24023e-14,
                           3.49837e-15, 3.51992e-15, 2.24845e-15, 1.34066e-15))

        hess_yl = hess_y - np.array(
            [3.47531e-11, 3.01035e-11, 1.61338e-11, 1.17782e-11, 6.91655e-12, 5.18683e-12, 2.57245e-12, 1.75349e-12,
             9.80935e-13, 4.16665e-13, 2.49006e-13, 6.43095e-14, 2.83361e-14, 5.3776e-15, 3.49837e-17, 3.51992e-17,
             2.24845e-17, 1.34066e-17])

        hess_yh = np.array(
            [4.92074e-11, 3.61898e-11, 1.93749e-11, 1.39342e-11, 8.2767e-12, 6.15261e-12, 3.14967e-12, 2.15466e-12,
             1.24357e-12, 5.73022e-13, 3.57712e-13, 1.20039e-13, 6.13372e-14, 2.27651e-14, 3.49837e-15, 3.51992e-15,
             2.24845e-15, 1.34066e-15]) - hess_y

        ax_flux.errorbar(x=hess_x * 1000, y=hess_y, yerr=[hess_yl, hess_yh], fmt=".", label="HESS 24.06.2014")

    if hess_flare:
        hess = pd.read_table("/home/michi/pyfact/hess2014flare.txt", names=["E", "F", "Fu", "Fl"])
        ax_flux.errorbar(x=hess.E.values*1000,
                         y=hess.F,
                         yerr=[(hess.F-hess.Fl).values, (hess.Fu-hess.F).values],
                         fmt=".",
                         label="Hess Flare 23.06.2014")

    if crab_do:
        crab_do_x_l = np.array(
            [251.18867, 398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0])
        crab_do_x_h = np.array(
            [398.1072, 630.9573, 1000.0, 1584.8929, 2511.8861, 3981.072, 6309.573, 10000.0, 15848.929])
        crab_do_x = pow(10, (np.log10(crab_do_x_l) + np.log10(crab_do_x_h)) / 2)
        crab_do_y = np.array(
            [7.600271e-10, 2.436501e-10, 6.902212e-11, 1.939978e-11, 5.119684e-12, 1.4333e-12, 3.756243e-13,
             1.220526e-13, 2.425761e-14])
        crab_do_y_err = np.array(
            [2.574657e-10, 3.90262e-11, 4.364535e-12, 1.083812e-12, 3.51288e-13, 1.207386e-13, 4.346247e-14,
             1.683129e-14, 6.689311e-15])
        ax_flux.errorbar(x=crab_do_x, y=crab_do_y, yerr=crab_do_y_err,
                         xerr=[crab_do_x - crab_do_x_l, crab_do_x_h - crab_do_x], fmt="o", label="Crab Dortmund")
    ax_flux.errorbar(x=bin_centers, y=flux, yerr=flux_err, xerr=energy_err, fmt=".", label=label, **kwargs)

    ax_sig.plot(bin_centers, significance, "o", **kwargs)

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    ax_flux.set_ylabel("Flux [$\mathrm{cm^{-2}TeV^{-1}s^{-1}}$]")
    ax_sig.set_xlabel("Energy [GeV]")
    ax_sig.set_ylabel('$S_{\mathrm{Li/Ma}} \,\, / \,\, \sigma$')
    ax_sig.grid()

    plt.setp(ax_flux.get_xticklabels(), visible=False)
    plt.legend()

    if filename is not None:
        fig.savefig(filename)

    return ax_sig, ax_flux


def plot_theta(thetasquare_binning, thetasquare_on, thetasquare_off, thetasq_cut, stats=None, filename=None):
    fig = plt.figure("ThetaSqare")
    ax = plt.subplot()
    ax.errorbar(x=(thetasquare_binning[1:]+thetasquare_binning[0:-1])/2,
                y=thetasquare_on,
                xerr=(thetasquare_binning[2]-thetasquare_binning[1])/2,
                yerr=np.sqrt(thetasquare_on),
                fmt=".",
                label="On Data")

    ax.errorbar(x=(thetasquare_binning[1:] + thetasquare_binning[0:-1]) / 2,
                y=0.2*thetasquare_off,
                xerr=(thetasquare_binning[2] - thetasquare_binning[1])/2,
                yerr=0.2*np.sqrt(thetasquare_off),
                fmt=".",
                label="0.2 * Off Data")

    ax.axvline(x=thetasq_cut, color='black', linestyle='-', label="Cut")
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(thetasquare_binning[0], thetasquare_binning[-1])

    plt.xlabel('$ \Theta^2 \, \mathrm{deg^2} $')
    plt.ylabel('Counts')
    plt.grid()

    if stats:
        # x_text = 0.2
        # y_limits = ax.get_ylim()
        # y_text = 0.5 * (y_limits[1]-y_limits[0]) + y_limits[0]

        text = "Overall Stats:\n"\
               "N_on: {0:.0f}".format(stats["n_on"]) + \
               "\nN_off: {0:.2f}".format(stats["n_off"]) + \
               "\nN_excess: {0:.2f}".format(stats["n_excess"]) + \
               "\nOn Time: {0:8.2f} h".format(stats["on_time_hours"]) + \
               "\n$\mathrm{\sigma_{LiMa}}$" + ": {0:3.2f}".format(stats["significance"])

        # plt.text(x_text, y_text, text)
        plt.plot([], [], ' ', label=text)
    plt.legend()

    if filename is not None:
        fig.savefig(filename)

    return ax
