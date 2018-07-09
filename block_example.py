import matplotlib.pyplot as plt
from blockspec.block.create_blocks_from_db import create_blocks
from blockspec.block.binning import multiple_day_binning

mrk = create_blocks(source="Mrk 501",
                    destination_path="/media/michi/523E69793E69574F/daten/Crab/blocks/",
                    use_thresh_and_hum=False,
                    dryrun=True,
                    prior=5.1,
                    time=0.5,
                    block_binning=multiple_day_binning)
plt.show()
print(mrk[0])
