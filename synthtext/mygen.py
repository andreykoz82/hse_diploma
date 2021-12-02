# %%
from itertools import chain

import random
import string
import os
import os.path as osp

from PIL import Image

import h5py
import pickle as cp

from synthgen import *
from common import *

# %%
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12,
             'D': 13,
             'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25,
             'Q': 26, 'R': 27,
             'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'Г': 36, 'о': 37, 'д': 38, 'е': 39, 'н': 40, ':': 41, '/': 42}


# %%
def exp_generator():
    month = ''.join(str(random.randint(0, 12)))
    if len(month) == 1:
        month = '0' + month
    year = ''.join(str(random.randint(2020, 2025)))
    return 'Годен до : ' + month + '/' + year


def sn_generator(size=13, repeat=100, chars=string.ascii_uppercase + string.digits):
    return ' '.join(''.join(random.choice(chars) for _ in range(size)) for _ in range(repeat)) + ' ' + exp_generator()


def gen(n=5000):
    for i in range(n):
        sn = sn_generator()
        yield sn


with open('data/newsgroup/newsgroup.txt', 'w') as f:
    for box in gen():
        f.write(box + '\n')

INSTANCE_PER_IMAGE = 3

im_dir = 'bg_img'
depth_db = h5py.File('depth.h5', 'r')
seg_db = h5py.File('seg.h5', 'r')

imnames = sorted(depth_db.keys())

RV3 = RendererV3('data', max_time=5)


def to_yolo(imgname, res, root_img, root_meta):
    for i, item in enumerate(res):
        Image.fromarray(item['img']).save(osp.join(root_img, f'{imgname}-{i}.png'))
        imh, imw = item['img'].shape[:2]

        chars = list(chain(*item['txt']))
        result = map(str.rstrip, chars)
        chars = list(filter(None, list(result)))

        assert len(chars) == item['charBB'].shape[-1]

        with open(osp.join(root_meta, f'{imgname}-{i}.txt'), 'w') as f:
            for char, charBB in zip(chars, item['charBB'].transpose(2, 0, 1)):
                x_min, y_min = charBB.min(axis=1)
                x_max, y_max = charBB.max(axis=1)
                x_center = .5 * (x_min + x_max) / imw
                y_center = .5 * (y_min + y_max) / imh
                width = (x_max - x_min) / imw
                height = (y_max - y_min) / imh
                f.write(f'{label_map[char]} {x_center} {y_center} {width} {height}\n')


for imname in sorted(os.listdir(im_dir)):
    if imname < 'hubble_4':
        continue
    print(f'start {imname} ...')
    if not any(imname.endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
        continue

    try:
        img = Image.open(osp.join(im_dir, imname)).convert('RGB')
    except:
        print(f'bad img {imname}')
        continue
    try:
        depth = depth_db[imname][:].T
    except:
        print(f'no depth for {imname}')
        continue
    depth = depth[:, :, 0]

    try:
        seg = seg_db['mask'][imname][:].astype('float32')
        area = seg_db['mask'][imname].attrs['area']
        label = seg_db['mask'][imname].attrs['label']
    except:
        print(f'no seg for {imname}')
        continue

    sz = depth.shape[:2][::-1]
    img = np.array(img.resize(sz, Image.ANTIALIAS))
    seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST)).astype('float32')

    try:
        res = RV3.render_text(img, depth, seg, area, label,
                              ninstance=INSTANCE_PER_IMAGE)
    except:
        print(f'{imname} problems while text rendering')
    if len(res) > 0:
        to_yolo(imname, res, 'results/images',
                'results/labels')

    print(f'end {imname}')
