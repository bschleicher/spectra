import ROOT
import sys
import os

result = ROOT.gSystem.Load('libmars.so')
if result != 0:
    raise ImportError(
        'Could not load libmars, Make sure to set your "LD_LIBRARY_PATH"'
    )

def check_file(entry):
    try:
        file = ROOT.TFile(entry)
        print("    File:", entry)
        for tree in trees:
            tree_object = file.Get(tree)
            n_entries = tree_object.GetEntries()
            print("        ", tree, " - Number of Entries:", n_entries)
            if (n_entries < 2):
                print("Less than 2 entries in Tree!")
                few_entries_list.append([entry, tree, n_entries])
    except:
        defect_list.append(entry)

def check_list(input_list):
    input_list = [entry.strip().replace(" ", "/") for entry in input_list if not entry.startswith('#')]
    for element in input_list:
        check_file(element)

    print("Files that raise error:")
    for entry in defect_list:
        print(entry)
    print("Files with less than 2 Entries:")
    for entry, tree, n_entries in few_entries_list:
        print(entry)
        print("In tree:", tree,"Entries:", n_entries)


if __name__ == '__main__':
    trees = ["Events", "Drive", "Rates"]
    defect_list=[]
    few_entries_list=[]

    directory_or_file = sys.argv[1]
    if directory_or_file.endswith(".txt"):
        star_list = list(open(directory_or_file, "r"))
        check_list(star_list)

    for file in os.listdir(directory_or_file):
        if file.endswith(".txt"):
            print(file)
            star_list = list(open(directory_or_file + "/" + file, "r"))
            check_list(star_list)
        elif file.endswith(".root"):
            print(file)
            check_file(directory_or_file + "/" + file)






