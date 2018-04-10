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
            if n_entries < 2:
                print("Less than 2 entries in Tree!")
                few_entries_list.append([entry, tree, n_entries])
        return True
    except AttributeError as inst:
        print(inst)
        print("File does probably not exist")
        defect_list.append(entry)
        return False


def check_txt(path, make_comment):
    with open(path, "r") as f:
        star_list = list(f)
    out_list = check_list(star_list)
    if make_comment == True:
        with open(path, "w") as f:
            f.write("".join(out_list))


def check_list(input_list):
    iter_list = [entry.strip().replace(" ", "/") for entry in input_list if not entry.startswith('#')]
    for n in range(len(iter_list)):
        check = check_file(iter_list[n])
        if not check:
            input_list[n] = "# " + input_list[n]
    print("Files that raise error:")
    for entry in defect_list:
        print(entry)
    print("Files with less than 2 Entries:")
    for entry, tree, n_entries in few_entries_list:
        print(entry)
        print("In tree:", tree, "Entries:", n_entries)
    return input_list


if __name__ == '__main__':
    trees = ["Events", "Drive", "Rates"]
    defect_list=[]
    few_entries_list=[]
    comment_defect_lines = False

    if len(sys.argv) == 2:
        directory_or_file = sys.argv[1]
    if len(sys.argv) == 3:
        directory_or_file = sys.argv[1]
        if sys.argv[2] == "1":
            comment_defect_lines = True
        print("Comment out runs that raise error")

    print(comment_defect_lines)

    if directory_or_file.endswith(".txt"):
        check_txt(directory_or_file, comment_defect_lines)

    for file in os.listdir(directory_or_file):
        if file.endswith(".txt"):
            print(file)
            check_txt(directory_or_file + "/" + file, comment_defect_lines)

        elif file.endswith(".root"):
            print(file)
            check_file(directory_or_file + "/" + file)






