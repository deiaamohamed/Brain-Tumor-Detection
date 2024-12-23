import os
import shutil
from sklearn.model_selection import train_test_split


brain_dir =r'D:\data\Brain'
no_brain_dir =r'D:\data\No_Brain'
#print(brain_dir)
# Paths for the new train/validation split
train_dir ='D:/data/train'
val_dir ='D:/data/val'
print("Brain files:", os.listdir(brain_dir))
print("No Brain files:", os.listdir(no_brain_dir))


# train/validation directories
os.makedirs(os.path.join(train_dir, 'Brain'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'No_Brain'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'Brain'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'No_Brain'), exist_ok=True)


# 20/100 * brain.size ... 20/100 * no brain.size
def split_and_move_files(src_dir, train_dst, val_dst, test_size=0.2):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    print("1 get all files done")

    train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)
    print("2 split done")

    for file in train_files:
        shutil.move(os.path.join(src_dir, file), os.path.join(train_dst, file))
    print("3 shutil done")

    for file in val_files:
        shutil.move(os.path.join(src_dir, file), os.path.join(val_dst, file))
    print("4 move done")

split_and_move_files(brain_dir, os.path.join(train_dir, 'Brain'), os.path.join(val_dir, 'Brain'))
print("done")
split_and_move_files(no_brain_dir, os.path.join(train_dir, 'No_Brain'), os.path.join(val_dir, 'No_Brain'))

print("Dataset split into training and validation sets.")
