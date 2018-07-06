# spectra
Tools and scripts to create spectra from FACT MARS analysis


Make a txt list of all runs and load star files from ISDC:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To download star files of Crab, Mrk 501 and Mrk 421 to your local hard disk do the following:

.. code::

   $ cd zd_spectra
   $ python make_list_of_runs.py -n runs.txt
   $ rsync -rv --files-from=runs.txt isdc:/gpfs0/fact/processing/data.r18753/ .

This will require about 101 GB of free space.
If you need different sources in specific timeranges use make_list_of_runs.py like this:

.. code::

    $ python make_list_of_runs.py -n runs.txt -b 20140113 -l 20141221 -s 'Crab' 'Mrk 501' '1ES 1959+650'

If you want all runs of all sources do:

.. code::

    $ python make_list_of_runs.py -n runs.txt -s 'None'