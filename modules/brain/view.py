import os
import sys
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'brain', 'images')

def find_nii_files(directory):
	patterns = ('*.nii', '*.nii.gz')
	files = []
	for p in patterns:
		files.extend(glob.glob(os.path.join(directory, p)))
	return sorted(files)

def load_image(path):
	img = nib.load(path)
	data = img.get_fdata()
	if data.ndim == 4:
		data = data[..., 0]
	return data

def show_slice(img, axis=2):
	if axis < 0 or axis >= img.ndim:
		raise ValueError('Invalid axis for image with ndim=%d' % img.ndim)
	idx = img.shape[axis] // 2
	if axis == 0:
		sl = img[idx, :, :]
	elif axis == 1:
		sl = img[:, idx, :]
	else:
		sl = img[:, :, idx]
	plt.figure(figsize=(6,6))
	plt.imshow(np.rot90(sl), cmap='gray')
	plt.title(f'Slice axis={axis} index={idx}')
	plt.axis('off')
	plt.show()

def main():
	if len(sys.argv) > 1:
		path = sys.argv[1]
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		if not os.path.exists(path):
			print(f'File not found: {path}')
			sys.exit(1)
	else:
		files = find_nii_files(IMAGES_DIR)
		if not files:
			print(f'No .nii or .nii.gz files found in {IMAGES_DIR}')
			sys.exit(1)
		path = files[0]
		if len(files) > 1:
			print(f'Found {len(files)} files, using: {path}')

	try:
		img = load_image(path)
	except Exception as e:
		print('Error loading image:', e)
		sys.exit(1)

	show_slice(img, axis=2)

if __name__ == '__main__':
	main()
