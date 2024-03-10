import glob, os
from math import *
from tqdm import tqdm
import shutil

input_folders = [
    'datasets/flowers/daisy/',
    'datasets/flowers/dandelion/',
    'datasets/flowers/roses/',
    'datasets/flowers/sunflowers/',
    'datasets/flowers/tulips/'
]

BASE_DIR_ABSOLUTE = 'D:\\Python\\PycharmProjects\\neiro\\FlowersNeiro'
OUT_DIR = './datasets/flowers_prepared/'

OUT_TRAIN = OUT_DIR + 'train/'
OUT_VAL = OUT_DIR + 'test/'

coeff = [80,20]
exceptions = ['classes']

if int(coeff[0]) + int(coeff[1]) > 100:
    print("Coeff can't exceed 100%.")
    exit(1)

def chunker(seq,size):
    return (seq[pos:pos + size] for pos in range(0,len(seq),size))

print(f"Preparing  images data by {coeff[0]/coeff[1]} rule.")
print(f"Sourse folders: {len(input_folders)} ")
print("Gathering data ...")

sourse = {}
for sf in input_folders:
    sourse.setdefault(sf,[])

    os.chdir(BASE_DIR_ABSOLUTE)
    os.chdir(sf)

    for filename in glob.glob("*.jpg"):
        sourse[sf].append(filename)

train = {}
val={}
for sk, sv in sourse.items():
    chunks = 10
    train_chunk = floor(chunks * (coeff[0]/ 100))
    val_chunk = chunks - train_chunk

    train.setdefault(sk, [])
    val.setdefault(sk, [])
    for item in chunker(sv, chunks):
        train[sk].extend(item[0:train_chunk])
        val[sk].extend(item[train_chunk:])

train_sum = 0
val_sum = 0

for sk,sv in train.items():
    train_sum+=len(sv)

for sk,sv in val.items():
    val_sum+=len(sv)

print(f'\nOverall TRAIN images count: {train_sum}')
print(f'Overall TEST images count: {val_sum}')

os.chdir(BASE_DIR_ABSOLUTE)
print("\nCoping TRAIN sourse items ot prepered folder ...")
for sk,sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_sourse = sk + item
        imgdile_dest = OUT_TRAIN+ sk.split('/')[-2] + '/'

        os.makedirs(imgdile_dest,exist_ok=True)
        shutil.copyfile(imgfile_sourse, imgdile_dest + item)

os.chdir(BASE_DIR_ABSOLUTE)
print('\nCopying VAL sourse items to prepared folder ...')
for sk,sv in tqdm(val.items()):
    for item in tqdm(sv):
        imgfile_sourse = sk + item
        imgdile_dest = OUT_VAL+ sk.split('/')[-2] + '/'

        os.makedirs(imgdile_dest,exist_ok=True)
        shutil.copyfile(imgfile_sourse, imgdile_dest + item)

print('\nDONE')
