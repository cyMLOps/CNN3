
# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'dataset_dogs_and_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
 # create label subdirectories
 labeldirs = ['dogs/', 'cats/']
 for labldir in labeldirs:
    newdir = dataset_home + subdir + labldir
    makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'train/'
for file in listdir(src_directory):
 src = src_directory + '/' + file
 dst_dir = 'train/'
 if random() < val_ratio:
    dst_dir = 'test/'
 if file.startswith('cat'):
    dst = dataset_home + dst_dir + 'cats/'  + file
    copyfile(src, dst)
 elif file.startswith('dog'):
    dst = dataset_home + dst_dir + 'dogs/'  + file
    copyfile(src, dst)

# create txt file on directory of training data
train_directory = dataset_home + subdirs[0]
image_list = []

for l in labeldirs:
    train_class_directory = train_directory + l
    for img in listdir(train_class_directory):
        image_list.append(train_class_directory + img)

with open('train_dataset.txt', 'w') as f:  
    for img in image_list:      
        f.write(img)
        f.write('\n')


print('------images created-----')