import os, shutil

# folder = "sketchy_dataset"
# mini_folder = "mini_sketchy_dataset"

# photo_folder = "sketchy_dataset\sketch"
# mini_photo_folder = "mini_sketchy_dataset\sketches"

# dirs = os.listdir(photo_folder)

# for dir_name in dirs:
#     print(dir_name)
#     path = os.path.join(photo_folder,dir_name)
#     # print(path)
#     files = os.listdir(path)
#     # print(files)
#     folder_path = os.path.join(mini_photo_folder, dir_name)
#     # print(folder_path)
#     os.mkdir(folder_path)
#     for i in range(5):
        # file_path = os.path.join(path, files[i])
        # shutil.copy(file_path, folder_path)

folder = "tenth_sketchy_data"
sketch_train_folder = "tenth_sketchy_data/train/sketch"
photo_train_folder = "tenth_sketchy_data/train/photo"

sketch_val_folder = "tenth_sketchy_data/val/sketch"
photo_val_folder = "tenth_sketchy_data/val/photo"

sketch_test_folder = "tenth_sketchy_data/test/sketch"
photo_test_folder = "tenth_sketchy_data/test/photo"

main_folder = "sketchy_dataset"
sketch_folder = "sketchy_dataset/sketch"
photo_folder = "sketchy_dataset/photos"
# os.mkdir(folder)


dirs = os.listdir(sketch_folder)

print(len(dirs))

os.makedirs(sketch_train_folder)
os.makedirs(sketch_test_folder)
os.makedirs(sketch_val_folder)
    

for dir_name in dirs:
    folder_path = os.path.join(sketch_folder, dir_name)
    files = os.listdir(folder_path)
    total = len(files)
    total = total//10
    split_1 = total//20
    split_2 = split_1
    split_3 = total - split_1 - split_2
    
    # print(total)
    # print(split_1)
    # print(split_2)
    # print(split_3)

    path = os.path.join(sketch_train_folder, dir_name)
    os.mkdir(path)
    for i in range(split_3):
        file_path = os.path.join(folder_path, files[i])
        shutil.copy(file_path, path)
    
    path = os.path.join(sketch_val_folder, dir_name)
    os.mkdir(path)
    for i in range(split_2):
        file_path = os.path.join(folder_path, files[i])
        shutil.copy(file_path, path)

    path = os.path.join(sketch_test_folder, dir_name)
    os.mkdir(path)
    for i in range(split_1):
        file_path = os.path.join(folder_path, files[i])
        shutil.copy(file_path, path)

dirs = os.listdir(sketch_train_folder)
print(len(dirs))
dirs = os.listdir(sketch_val_folder)
print(len(dirs))
dirs = os.listdir(sketch_test_folder)
print(len(dirs))

    

