import matplotlib.pyplot as plt
from blockspec.block.create_blocks_from_db import create_blocks
from blockspec.block.binning import multiple_day_binning
from blockspec.block import BlockAnalysis


mrk = create_blocks(source="Mrk 501",
                    destination_path="/media/michi/523E69793E69574F/daten/Testing/blocks/",
                    start=20170505,
                    stop=20170707,
                    use_thresh_and_hum=False,
                    dryrun=False,
                    prior=5.1,
                    time=0.5,
                    block_binning=multiple_day_binning)
plt.show()
print(mrk[0])


ceres_list = []
for i in range(8):
    ceres_list.append("/home/michi/read_mars/ceres_part" + str(i) + ".h5")

blocks = BlockAnalysis(
                       ceres_list=["/home/michi/ceres.h5"],
                       ganymed_mc="/media/michi/523E69793E69574F/daten/gamma/hzd_gammasall-analysis.root",
                       basepath='/media/michi/523E69793E69574F/daten/Testing/',
                       ganymed_path="/media/michi/523E69793E69574F/gamma/ganymed.C",
                       mars_directory="/home/michi/Mars/",
                       spec_identifier="forest2_",
                       source_name="Mrk 501")


