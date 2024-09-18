import os


def change_extensions(path, new_extension):
    for old_filename in os.listdir(path):

        os.rename(os.path.join(path, old_filename), os.path.join(path,old_filename[:-3]+new_extension))
        print(old_filename[:-3]+new_extension)


change_extensions("data/images/train", "jpg")
