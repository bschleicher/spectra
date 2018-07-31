# spectra
Tools and scripts to create spectra from FACT MARS analysis



1. Prepare Gamma Simulations:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the effective area, we need the information about all simulated events.
This information is stored in ceres-files.
For the effective area, all simulated events passing the detector simulation and the same analysis as data events
are needed. This information can be obtained from a ganymed-file. In order to run your ganymed on the data, you need
the star files.
Here, it is important to take the appropriate ones. For the ISDC analysis, this is the one stored in the star_new folder.


If you want, copy all ceres files to your hard drive:

.. code::

    $ rsync -rv --include="*/" --include="gamma_*/*/*_I_*.root.gz" --exclude="*" isdc:/gpfs0/fact/monte-carlo/dortmund/ceres/ /media/michi/523E69793E69574F/gamma/ceres/

In case you have to unzip the ceres files just do
    $ cd ceres
    $ gzip -d */*/*.root.gz

The same for the star files:

.. code::

    $ rsync -rv --include="*/gamma_be*/" --include="gamma_bernd*/*/star.root" --exclude="*" isdc:/gpfs0/fact/monte-carlo/dortmund/star_new/ /media/michi/523E69793E69574F/gamma/star_new/


The next step is to make text files with one line for each file to be analyzed. To make such a list for the ceres files,
I suggest you cd to your ceres folder and do

.. code::

    $ cd ceres
    $ find /media/michi/523E69793E69574F/gamma/ceres -type f -size +4096c -iname *.root | sort > ceres_files.txt

For the star files, the procedure is more annoying, because for some strange reason, the convention is to use a blank
space for the last / in the filepath. The easiest way for me was to use sed as follows.
The 8 in the command specifies, that only the 8. occurence of / is changed. You have to adapt this number to the ones of
your path. Be careful with the -i option, because it changes the file in place.

.. code::

    $ cd ..
    $ cd star_new
    $ find /media/michi/523E69793E69574F/gamma/star_new -type f -size +4096c -iname star.root | sort > star_files.txt
    $ sed -i 's_/_ _8' star_files.txt

in case you need to undo the exchange of / with a blank space, you can do, where g stands for global, what changes all
occurences.

.. code::

    $ sed -i 's__/_g' star_files.txt

1.1 Store info of 20000 ceres files in 8 hdf files, for significant speedup.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following python codesnippet shows how to store the info of all ceres files to 8 hdf files stored in the
current directory.

. code-block:: python

    from tqdm import tqdm
    from read_mars import read_mars

    cereslist = "/media/michi/523E69793E69574F/gamma/ceres/ceres_files.txt"

    callisto_list = list(open(cereslist, "r"))

    x = read_mars(callisto_list[0].strip(), tree="OriginalMC")

    length = len(callisto_list)
    n_file = len(x)
    df = pd.DataFrame(index=range(n_file*length), columns=x.columns, dtype=np.float32)
    for this in tqdm(range(length)):
        entry = callisto_list[this].strip()
        lowindex = this*n_file
        highindex = (this+1)*n_file
        df.iloc[lowindex:highindex]  = read_mars(entry, tree="OriginalMC", leaf_names=leafs).values


    length = len(df)
    n_parts = 8
    divisor = length//n_parts

    for n in tqdm(range(n_parts)):
        part = df.iloc[(n*divisor): (1+n)*divisor]
        part.to_hdf("ceres_part" + str(n) + ".h5", "table")

1.2 Run Ganymed for Monte Carlos.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Go to the Mars directory and do

.. code::
    $ cd
    $ cd Mars
    $ .x /media/michi/523E69793E69574F/gamma/ganymed.C("/media/michi/523E69793E69574F/gamma/star_new/star_files.txt","/media/michi/523E69793E69574F/gamma/star_new/gammasall",0,0,1)

3. Get the Data: Make a txt list of all runs and load star files from ISDC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To download star files of Crab, Mrk 501 and Mrk 421 to your local hard disk do the following:

.. code::

   $ cd zd_spectra
   $ python make_list_of_runs.py -n runs.txt
   $ rsync -rv --files-from=runs.txt isdc:/gpfs0/fact/processing/data.r18753/ .

This will require about 101 GB of free space.
runs.txt will contain entries with one run per line: star/2014/12/21/20141221_123_I.root

If you need different sources in specific timeranges for a different folder, use make_list_of_runs.py like this:

.. code::

    $ python make_list_of_runs.py -n runs.txt -f 20140113 -l 20141221 -s 'Crab' 'Mrk 501' '1ES 1959+650' -b /path/star/

If you want all runs of all sources do:

.. code::

    $ python make_list_of_runs.py -n runs.txt -s 'None'
