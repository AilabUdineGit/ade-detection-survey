import os

def recursive_check_path(path):
    folders = path.split("/")
    for folder_idx in range(len(folders)):
        partial_path = "/".join(folders[:folder_idx+1])+"/"
        check_path(partial_path.replace("//","/"))

def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)