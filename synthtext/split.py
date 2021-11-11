# %%
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

images = sorted(os.listdir('/SynthText/results/images'))
labels = sorted(os.listdir('/SynthText/results/labels'))

train_img, val_img = train_test_split(images, test_size=0.2)

source_path = '/SynthText/results/'
dest_path = '/YoloOCR/dataset/'
# %%
train_lbl = []
val_lbl = []
for file in train_img:
    train_lbl.append(file[:-4] + '.txt')

for file in val_img:
    val_lbl.append(file[:-4] + '.txt')

# %%
for image, label in zip(train_img, train_lbl):
    copyfile(source_path + 'images/' + image, dest_path + 'train/' + 'images/' + image)
    copyfile(source_path + 'labels/' + label, dest_path + 'train/' + 'labels/' + label)

for image, label in zip(val_img, val_lbl):
    copyfile(source_path + 'images/' + image, dest_path + 'val/' + 'images/' + image)
    copyfile(source_path + 'labels/' + label, dest_path + 'val/' + 'labels/' + label)
