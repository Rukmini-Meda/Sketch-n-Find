import os
import shutil
import gdown
from zipfile import ZipFile

URL = "https://drive.google.com/u/0/uc?id=0B7ISyeE8QtDdTjE1MG9Gcy1kSkE"
OUTPUT_FILE = "dataset.zip"
# Give the name of the directory in sketchy dataset which contains the sketches you want to train.
ID = "tx_000000000000"
SKETCH_DIR = os.path.join(OUTPUT_FILE[: len(OUTPUT_FILE) - 4], "256x256", "sketch", ID)
PHOTO_DIR = os.path.join(OUTPUT_FILE[: len(OUTPUT_FILE) - 4], "256x256", "photo", ID)
# DATA_DIR = ""

# dir_list = [dir_name for dir_name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_name))]

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

def rename_files(source_path):

    for count, filename in enumerate(os.listdir(source_path)):
        dst = str(count+1)+"funny.png"
        os.rename(os.path.join(source_path,filename),os.path.join(source_path,dst))
    for count, filename in enumerate(os.listdir(source_path)):
        dst = str(count+1)+".png"
        os.rename(os.path.join(source_path,filename),os.path.join(source_path,dst))

# copy_images_to_train_data()
# copy_labels_to_file()

def download_extract_sketchy_dataset():
    
    print(f"Downloaded and extracted files will be in \"{OUTPUT_FILE[:len(OUTPUT_FILE) - 4]}\" folder")

    gdown.download(URL, OUTPUT_FILE, quiet=False)

    with ZipFile(OUTPUT_FILE, "r") as zipobj:
        zipobj.extractall(OUTPUT_FILE[:len(OUTPUT_FILE) - 4])

    output_folder = OUTPUT_FILE[:len(OUTPUT_FILE) - 4]
    return output_folder, ID

class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, checkpoint_dir):
    file_name = 'last.pth.tar'
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, file_name))

def load_checkpoint(checkpoint, image_model, sketch_model, domain_model=None, optimizer=None):
    if not os.path.exists(checkpoint):
        raise Exception("File {} doesn't exist".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    print('Loading the models from the end of net iteration %d' % (checkpoint['iteration']))
    image_model.load_state_dict(checkpoint['image_model'])
    sketch_model.load_state_dict(checkpoint['sketch_model'])
    if domain_model: domain_model.load_state_dict(checkpoint['domain_model'])
    if optimizer:
      optimizer.load_state_dict(checkpoint['optim_dict'])

def get_sketch_images_grids(sketches, images, similarity_scores, k, num_display):

  if num_display == 0 or k == 0:
    return None, None
  num_sketches = sketches.shape[0]
  indices = np.random.choice(num_sketches, num_display)

  cur_sketches = sketches[indices]; cur_similarities = similarity_scores[indices]
  top_k_similarity_indices  = np.flip(np.argsort(cur_similarities, axis = 1)[:, -k:], axis = 1).copy()
  top_k_similarity_values = np.flip(np.sort(cur_similarities, axis = 1)[:,-k:], axis = 1).copy()
  matched_images = [images[top_k_similarity_indices[i]] for i in range(num_display)]

  list_of_sketches = [np.transpose(cur_sketches[i].cpu().numpy(), (1,2,0)) for i in range(num_display)]
  list_of_image_grids = [np.transpose(make_grid(matched_images[i], nrow = k).cpu().numpy(), (1,2,0)) for i in range(num_display)]

  return list_of_sketches, list_of_image_grids
