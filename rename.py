import os


def rename(path:str):
    files = os.listdir(path)

    for file in files:
        filepath = path + "/" + file

        newfile = file.replace("æ", "ae").replace("ø", "oe").replace("å", "aa").replace("ü", "u")

        new_filepath = path + "/" + newfile
        os.rename(filepath, new_filepath)


rename("F:\Dataset\FoodX\labels")
