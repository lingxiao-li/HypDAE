import os
from torch.utils.data import Dataset
from PIL import Image
import random
import sys
import cv2
sys.path.append(".")
from configs import data_configs
import numpy as np
"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def find_classes(directory):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ImagesDataset(Dataset):

	def __init__(self, dataset_type):
		if dataset_type not in data_configs.DATASETS.keys():
				Exception(f'{dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {dataset_type}')
		self.dataset_type = dataset_type
		dataset_args = data_configs.DATASETS[dataset_type]
		transforms_dict = dataset_args['transforms']().get_transforms()
		source_root = dataset_args['train_source_root']
		target_root = dataset_args['train_target_root']
		target_transform = transforms_dict['transform_gt_train']
		source_transform = transforms_dict['transform_source']
		train_transform = transforms_dict['transform_train']
		self.source_paths = sorted(make_dataset(source_root))
		self.target_paths = sorted(make_dataset(target_root))
		self.classes, self.class_to_idx = find_classes(source_root)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.train_transform = train_transform

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		while(True):
			try: 
				item = self.get_sample(index)
				return item
			except:
				index = np.random.randint(0, len(self.source_paths)-1)
    
    
	def detele_iccfile(image_path):
		img = Image.open(image_path)
		img.info.pop('icc_profile', None)
		img.save(image_path)
        

	def get_sample(self, index):
		from_path = self.source_paths[index]
		# if self.dataset_type == 'nabirds_encode':
		# 	self.detele_iccfile(from_path)
		class_label = from_path.split('/')[-2]
		label = self.class_to_idx[class_label]
		from_im = cv2.imread(from_path)
		from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)

		# from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
		# from_im = Image.open(from_path)
		# from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		# if self.dataset_type == 'nabirds_encode':
		# 	self.detele_iccfile(to_path)
		to_im = cv2.imread(to_path)
		to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
		# to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
		# to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(image=to_im)["image"]
			to_im = np.array(to_im).astype(np.uint8)
			to_im = (to_im / 127.5 - 1.0).astype(np.float32)
			# to_im = Image.fromarray(to_im["image"])

		if self.source_transform:
			from_im = self.source_transform(image=from_im)["image"]
			from_im = np.array(from_im).astype(np.uint8)
			from_im = (from_im / 127.5 - 1.0).astype(np.float32)
			# from_im = Image.fromarray(from_im["image"])
		else:
			from_im = to_im
   
		return {"ref": from_im, "jpg": to_im}
