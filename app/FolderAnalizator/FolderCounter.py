import os

def get_tree_dir(adres):
    return depth_search_folder(adres)

def get_tree_images(adres):
    return os.listdir(adres)


def depth_search_folder(adres_folder :str):
    print(f'depth_search_folder: {os.listdir(adres_folder)}')

    _images_tree = os.listdir(adres_folder)
    num = 0

    for obj in _images_tree:
        print(f'depth_search_folder: {abs_adres(adres_folder, obj)}  {os.path.isdir(abs_adres(adres_folder, obj))}')
        if os.path.isdir(abs_adres(adres_folder, obj)):
            _images_tree[num] = [_images_tree[num]]
            _images_tree[num].append(depth_search_folder(adres_folder + "/" + obj))

        num+=1

    list(filter(None, _images_tree))
    return _images_tree

def abs_adres(adres_folder :str, obj):
    return f'{os.getcwd().replace("app","")}/{adres_folder.replace("../","")}/{obj}'

def get_count_folders_files_aderss(folder :str):
    return os.listdir(folder)