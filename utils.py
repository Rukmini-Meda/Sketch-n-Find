import os
import shutil

# Give the name of the directory in sketchy dataset which contains the sketches you want to train.
DATA_DIR = "./tx_000000000000"
dir_list = [dir_name for dir_name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_name))]

def copy_images_to_train_data():

    count = 0
    for dir_name in dir_list:
        dir_path = os.path.join(DATA_DIR, dir_name)
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            copied_file_path = shutil.copy(file_path, "./train/data/"+str(count+1)+".png")
            print(copied_file_path)
            count += 1

def copy_labels_to_file():

    with open("./train/labels.txt", "a") as f:
        for dir_name in dir_list:
            files = os.listdir(os.path.join(DATA_DIR, dir_name))
            num = len(files)
            for i in range(num):
                f.write("\n")
                f.write(dir_name)

    with open("./train/labels.txt", "r") as f:
        print(f.read())

def rename_files():

    source_path = "./train/data"

    for count, filename in enumerate(os.listdir(source_path)):
        dst = str(count+1)+"funny.png"
        os.rename(os.path.join(source_path,filename),os.path.join(source_path,dst))
    for count, filename in enumerate(os.listdir(source_path)):
        dst = str(count+1)+".png"
        os.rename(os.path.join(source_path,filename),os.path.join(source_path,dst))

# copy_images_to_train_data()
# copy_labels_to_file()