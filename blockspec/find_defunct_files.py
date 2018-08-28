import ROOT
import sys
import os

_result = ROOT.gSystem.Load('libmars.so')
if _result != 0:
    print(
        'Could not load libmars, make sure to set your "LD_LIBRARY_PATH". If it is set, it was loaded before.'
    )


def _check_file(entry, trees=None, defect_list=None, few_entries_list=None):
    if trees is None:
        trees = ["Events", "Drive", "Rates"]
    try:
        file = ROOT.TFile(entry)
        print("    File:", entry)
        for tree in trees:
            tree_object = file.Get(tree)
            n_entries = tree_object.GetEntries()
            print("        ", tree, " - Number of Entries:", n_entries)
            if n_entries < 2:
                print("Less than 2 entries in Tree!")
                few_entries_list.append([entry, tree, n_entries])
        return True, defect_list, few_entries_list
    except AttributeError as inst:
        print(inst)
        print("File does probably not exist")
        defect_list.append(entry)
        return False, defect_list, few_entries_list


def _check_txt(path, make_comment, defect_list, few_entries_list, trees=None):
    with open(path, "r") as f:
        star_list = list(f)
    out_list, defect_list, few_entries_list = _check_list(star_list, defect_list, few_entries_list, trees=trees)
    if make_comment is True:
        with open(path, "w") as f:
            f.write("".join(out_list))
    return defect_list, few_entries_list


def _check_list(input_list, defect_list, few_entries_list, trees=None):
    iter_list = [entry.strip().replace(" ", "/") for entry in input_list if not entry.startswith('#')]
    for n in range(len(iter_list)):
        check, defect_list, few_entries_list = _check_file(iter_list[n],
                                                           defect_list=defect_list,
                                                           few_entries_list=few_entries_list,
                                                           trees=trees)
        if not check:
            input_list[n] = "# " + input_list[n]
    print("Files that raise error:")
    for entry in defect_list:
        print(entry)
    print("Files with less than 2 Entries:")
    for entry, tree, n_entries in few_entries_list:
        print(entry)
        print("In tree:", tree, "Entries:", n_entries)
    return input_list, defect_list, few_entries_list


def check_mars_files(directory_or_file, comment_defect_lines=True, trees=None):
    """Check a root file, a directory containing root files and txt runlists or a txt runlist file if they contain
       events in some trees. 
       If trees=None (default), ["Events", "Drive", "Rates"] will be checked. This is good to check Star files."""

    defect_list = []
    few_entries_list = []

    if directory_or_file.endswith(".txt"):
        defect_list, few_entries_list = _check_txt(directory_or_file,
                                                   comment_defect_lines,
                                                   defect_list,
                                                   few_entries_list,
                                                   trees=trees)

    elif directory_or_file.endswith(".root"):
        check, defect_list, few_entries_list = _check_file(directory_or_file, defect_list=defect_list,
                                                           few_entries_list=few_entries_list, trees=trees)

    else:
        for file in os.listdir(directory_or_file):
            if file.endswith(".txt"):
                print(file)
                defect_list, few_entries_list = _check_txt(directory_or_file + "/" + file,
                                                           comment_defect_lines,
                                                           defect_list,
                                                           few_entries_list,
                                                           trees=trees)

            elif file.endswith(".root"):
                print(file)
                check, defect_list, few_entries_list = _check_file(directory_or_file + "/" + file,
                                                                   defect_list=defect_list,
                                                                   few_entries_list=few_entries_list,
                                                                   trees=trees)
    return defect_list, few_entries_list


if __name__ == '__main__':

    import argparse

    p = argparse.ArgumentParser(description="Check existence and entries of Mars root files")
    p.add_argument("directory_or_file", help="Directory or txt containing one root file per line")
    p.add_argument("-c", "--comment_defect_lines",
                   default=False,
                   type=bool,
                   help="If true, comment out the line of the defect or missing root files in a txt ")
    args = p.parse_args()

    if args.comment_defect_lines:
        print("Commenting out runs that raise error")

    check_mars_files(args.directory_or_file, args.comment_defect_lines)
