import ROOT
import sys

result = ROOT.gSystem.Load('libmars.so')
if result != 0:
    raise ImportError(
        'Could not load libmars, Make sure to set your "LD_LIBRARY_PATH"'
    )


if __name__ == '__main__':
    star_list = list(open(sys.argv[1], "r"))
    star_list = [entry.strip().replace(" ", "/") for entry in star_list if not entry.startswith('#')]
    trees = ["Events", "Drive", "Rates"]
    defect_list=[]
    few_entries_list=[]


    for entry in star_list:
        try:
            file = ROOT.TFile(entry)
            print("    File:",entry)
            for tree in trees:
                tree_object = file.Get(tree)
                n_entries = tree_object.GetEntries()
                print("        ", tree, " - Number of Entries:", n_entries)
                if (n_entries < 2):
                    print("Less than 2 entries in Tree!")
                    few_entries_list.append([entry, tree, n_entries])
        except:
            defect_list.append(entry)
    print("Files that raise error:")
    for entry in defect_list:
        print(entry)
    print("Files with less than 2 Entries:")
    for entry, tree, n_entries in few_entries_list:
        print(entry)
        print("In tree:", tree,"Entries:", n_entries)

