from spectrum_class import Spectrum
import matplotlib.pyplot as plt

# Monte Carlo simulated events:
ceres_list = []

for i in range(8):
    ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")

# Monte Carlo surviving events:

ganymed_mc = "/media/michi/523E69793E69574F/daten/gamma/hzd_gammasall-analysis.root"

# Star files of data:

star_files = ["/media/michi/523E69793E69574F/daten/crab2.txt"]
star_list = []
for entry in star_files:
    star_list += list(open(entry, "r"))

# Ganymed result:
ganymed_result = "/media/michi/523E69793E69574F/daten/crab2-analysis.root"


spectrum = Spectrum(run_list_star=star_list,
                    ganymed_file_data=ganymed_result,
                    list_of_ceres_files=ceres_list,
                    ganymed_file_mc=ganymed_mc)
# spectrum.set_correction_factors()
spectrum.optimize_theta()
spectrum.optimize_ebinning()

spectrum.calc_differential_spectrum()

spectrum.plot_flux()
spectrum.plot_thetasq()
path = "/media/michi/523E69793E69574F/xy.json"
spectrum.save(path)

plt.show()

second = Spectrum()
second.load(path)

second.plot_flux(crab_do=True)
second.plot_thetasq()

plt.show()
