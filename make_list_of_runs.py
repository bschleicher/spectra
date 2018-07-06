from fact.qla import get_qla_data

from blockspec.block.create_blocks_from_db import save_runlist_star


if __name__ == '__main__':

    import argparse

    p = argparse.ArgumentParser(description="Create text file of runs retrieved from the database.")
    p.add_argument("-n",
                   "--filename",
                   help="Name of the file to store the list of runs into",
                   required=True)
    p.add_argument("-b",
                   "--basepath",
                   default="star/",
                   help="Basepath of file structure.")
    p.add_argument("-s",
                   "--sources",
                   default=["Crab", "Mrk 501", "Mrk 421"],
                   nargs="+",
                   help="List of sources to get runs for. If 'None' return runs for all sources. " +
                        "Usage: -s 'Mrk 501' 'Crab'")
    p.add_argument("-f",
                   "--first_night",
                   default=None,
                   type=int,
                   help="Date of the first night to retrieve data from must be in from of 20150224")
    p.add_argument("-l",
                   "--last_night",
                   default=None,
                   type=int,
                   help="Date of the last night to retrieve data from must be in from of 20150224")
    args = p.parse_args()

    for source in args.sources:
        if source == 'None':
            args.sources = None

    df = get_qla_data(first_night=args.first_night,
                      last_night=args.last_night,
                      sources=args.sources)
    save_runlist_star(args.filename,
                      df,
                      args.basepath,
                      use_ganymed_convention=False)
