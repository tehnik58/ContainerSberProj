from FolderAnalizator.FolderCounter import get_tree_images
from FolderAnalizator.FolderCounter import get_tree_dir


class Folder_Analizator:
    folder_with_img = []

    def pain(self, mas, deep = 0, name =''):
        for a in mas:
            if type(a[1]) == list:
                print(f'{name}/{a[0]}')
                if self.pain(a[1], deep+1, name + '/' + a[0]):
                    self.folder_with_img.append([f'{name}/{a[0]}', get_tree_images(f'{name}/{a[0]}')])
                    #Psmain.analize_folder(a[1], f'{name}/{a[0]}')
            else:
                return True

    def analyze_subfolders_starting_from(self, folder_adress):
        self.folder_with_img.clear()

        a = get_tree_dir(folder_adress)
        print(f'a: {a}')
        self.pain(a, name=  folder_adress)

        return self.folder_with_img