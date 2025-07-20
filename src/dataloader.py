import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import load_image, train_pca
from sklearn.decomposition import PCA
import abc

class HyperspectralDataset(abc.ABC, Dataset):
    """
    This class assumes that the input image is a single large image and the labels are in the form of patches.
    The patches are created by sliding a window of size patch_size over the image with a stride of stride.
    
    """        
    def __init__(self, data_config, inflate_3d=True, transform=None, train=True, train_p=0.8):
        
        self.paths_file = data_config['paths_file'] # Paths file contains "image_path, label_path" pairs
        self.metadata_file = data_config['metadata_file']
        self.shuffle = data_config['shuffle']
        self.patch_size = data_config['patch_size']
        self.stride = data_config['stride']
        self.in_bands = data_config['in_bands']
        self.whiten = data_config['whiten']
        self.train_p = train_p
        self.transform = transform
        self.image_paths = [] 
        self.label_paths = []
        #self.labels = []
        self.inflate_3d = inflate_3d
        self.train = train
        # Metadata contains information about the image size, crop, and class names
        self.name_to_id = {}
        self.id_to_name = {}
        self.crop = None
        self.size = None
        self.parse_metadata()
        
            
        with open(self.paths_file) as f:
            for line in f:
                img_path, label_path = line.strip().split(',')
                self.image_paths.append(img_path)
                self.label_paths.append(label_path)
                
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("Number of images and labels must be the same")
        if len(self.image_paths) == 0:
            raise ValueError("No images found")
        
        # Delete these lines with caution, feature not guaranteed to work
        if len(self.image_paths) != 1:
            raise NotImplementedError("Only one image is supported")
        
        # load the image and preprocess it
        self.image, _ = load_image(self.image_paths[0])
        self.image = (self.image - self.image.min(axis=(0, 1), keepdims=True)) / \
                     (self.image.max(axis=(0, 1), keepdims=True) - self.image.min(axis=(0, 1), keepdims=True))
                     
        if self.in_bands != -1 and not self.in_bands == self.image.shape[2]:
            flat_data = self.image.reshape(-1, self.image.shape[-1])

            assert self.in_bands > 0
            print(f"Performing PCA transformation on input image, scaling down to {self.in_bands} bands")
            pca = PCA(n_components=self.in_bands, whiten=self.whiten)
            flat_data = pca.fit_transform(flat_data)
            self.image = flat_data.reshape(self.image.shape[0], self.image.shape[1], -1)
        
        # load the PCA module for visualization
        self.pca_module = train_pca(self.image, n_components=3)
        self.image = torch.from_numpy(self.image).float()

        # crop according to the metadata
        self.image = self.image[self.crop[2]:self.crop[3], self.crop[0]:self.crop[1]]
        self.image = self.image.permute(2, 0, 1) # <channels>, <height>, <width>
        
        print(f'Input image full shape: {self.image.shape}')

        # labels are loaded by the child class
        self.patches = None
        self.patch_indices = None
                
    def build(self):
        """Call abstract method to construct patches, split and shuffle
        """
        print(f"Building dataset of type {self.__class__.__name__}")
        
        self.patches, self.patch_indices = self.construct_patches() # abstract method to be implemented in the child class
                
        if self.shuffle:
            indices = torch.randperm(len(self.patches))
            self.patches = [self.patches[i] for i in indices]
            self.patch_indices = [self.patch_indices[i] for i in indices]
                
        if self.train:
            self.patches = self.patches[:int(self.train_p * len(self.patches))]
        else:
            self.patches = self.patches[int(self.train_p * len(self.patches)):]
        
    def parse_metadata(self):
        with open(self.metadata_file) as f:
            for line in f:
                key, value = line.strip().split(':')
                if key == 'pair':
                    id, name = value.split(',')
                    self.id_to_name[int(id)] = name
                    self.name_to_id[name] = int(id)
                elif key == 'crop':
                    x_start, x_end, y_start, y_end = value.split(',')
                    x_start = int(x_start)
                    y_start = int(y_start)
                    if x_end == 'None':
                        x_end = None
                    else:
                        x_end = int(x_end)
                    if y_end == 'None':
                        y_end = None
                    else:
                        y_end = int(y_end)
                    self.crop = (x_start, x_end, y_start, y_end)
                elif key == 'size':
                    height, width = value.split(',')
                    self.size = (int(height), int(width))
    
        assert self.size is not None, "Size not found in metadata"
        assert self.crop is not None, "Crop not found in metadata"
        assert len(self.id_to_name) > 0, "No class names found in metadata"
                    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        l = self.patches[idx] # <height, width, classes>

        # get corresponding patch of the image to idx
        (start_i, start_j), (end_i, end_j) = self.patch_indices[idx]
        i = self.image[:, start_i:end_i, start_j:end_j] # <channels>, <height>, <width>
          
        if self.transform:
            l = l.permute(2, 0, 1) # <classes>, <height>, <width>
            l = l.float() # <classes>, <height>, <width>
            i, l = self.transform(i, l)
            l = l.permute(1, 2, 0) # <height>, <width>, <classes>
            
        
        if self.inflate_3d:
            i = i.unsqueeze(0) # channels, depth (bands), height, width
        
        #l = l.reshape(-1, l.shape[2]) # <height * width, classes>
        return i, l

    def pca_transform_patch(self, patch):
        if isinstance(patch, torch.Tensor):
            patch = patch.clone().cpu().numpy()
        elif isinstance(patch, np.ndarray):
            patch = patch.copy()
        
        # Apply PCA transformation to the patch to 3 bands
        patch = patch.squeeze(0) # <bands, height, width>
        patch = patch.transpose(1, 2, 0) # <height>, <width>, <bands>
        patch = patch.reshape(-1, patch.shape[2]) # <height * width>, <bands>
        patch = self.pca_module.transform(patch)
        patch = patch.reshape(self.patch_size[0], self.patch_size[1], -1) # <height>, <width>, 3
        patch = patch.transpose(2, 0, 1) # 3, <height>, <width>
        return patch

    def construct_patches(self):
        """Abstract method to be implemented in the child class
        """
        raise NotImplementedError("construct_patches() must be implemented in the child class")
        
class NumpyDataset(HyperspectralDataset):
    def __init__(self, data_config, inflate_3d=True, transform=None, train=True):
        super().__init__(data_config, inflate_3d=inflate_3d, transform=transform, train=train)
        #self.image_paths = [data_config['image_path']]   
    
    def construct_patches(self):
        labels = []
        for label in self.label_paths:
            labels.append(torch.load(label))
        
        self.patches = []
        self.patch_indices = [] # (start, end) indices of patches for each image

        for label in labels:
            # label is shape <height, width, classes>
            start = len(self.patches)
            for i in range(0, label.shape[0] - self.patch_size[0], self.stride):
                for j in range(0, label.shape[1] - self.patch_size[1], self.stride):
                    patch = label[i:i+self.patch_size[0], j:j+self.patch_size[1], :] # The last patch may be smaller than the patch size
                    self.patches.append(patch)
                    self.patch_indices.append(((i, j), (i+self.patch_size[0], j+self.patch_size[1])))     
        
        # remove the last patch since its not guaranteed to be of the correct size
        del self.patches[-1]
        del self.patch_indices[-1]
        assert len(self.patches) == len(self.patch_indices) and len(self.patches) > 0, "Unexpected No. of patches found"
        
    
        return self.patches, self.patch_indices
    
class PolygonDataset(HyperspectralDataset):
    def __init__(self, data_config, inflate_3d=True, transform=None, train=True):
        super().__init__(data_config, inflate_3d=inflate_3d, transform=transform, train=train)
        assert self.crop == (0, None, 0, None), "Crop found in metadata; not supported for polygon dataset"
        
    def construct_patches(self):
        # Polygon file is output from labelme
        """ "shapes": [
    {
      "label": "Umbrella Tree",
      "points": [
          [
          293.09090909090907,
          5.045454545454607
        ],
        [
          247.63636363636363,
          150.50000000000006
        ], ...
        """
        self.patches = []
        self.patch_indices = []
        
        # get the polygons and slide window into patches
        for label_path in self.label_paths:
            self.label_path = label_path
            self._parse_label_file()

        return self.patches, self.patch_indices
    
    def _parse_label_file(self):
        with open(self.label_path) as f:
            polygons = json.load(f)
            polygons = polygons['shapes']
            for polygon in polygons:
                label = polygon['label']
                points = np.array(polygon['points'], np.int32)
                # get the bounding box of the polygon
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                # get the patch from the image
                for i in range(int(x_min), int(x_max), self.stride):
                    for j in range(int(y_min), int(y_max), self.stride):
                        x_patch_min = i
                        x_patch_max = i + self.patch_size[0]
                        y_patch_min = j
                        y_patch_max = j + self.patch_size[1]
                        image_patch = self.image[:, x_patch_min:x_patch_max, y_patch_min:y_patch_max]
                        # make sure the image will have enough pixels to be a patch
                        if image_patch.shape[1] != self.patch_size[0] or image_patch.shape[2] != self.patch_size[1]:
                            print(f"Skipping patch {x_patch_min}, {y_patch_min}, {x_patch_max}, {y_patch_max} because it will be out of bounds")
                            continue
                        
                        # check if at least 90% of the patch is inside the polygon
                        poly_mask = np.zeros((self.patch_size[0], self.patch_size[1]), dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [points - (x_patch_min, y_patch_min)], self.name_to_id[label])
                        intersection = (poly_mask != 0).sum() / (self.patch_size[0] * self.patch_size[1])
                        if intersection < 0.85: 
                            continue
                        
                        # create the label for the patch
                        patch = torch.zeros(self.patch_size[0], self.patch_size[1], len(self.id_to_name))
                        patch[:, :, self.name_to_id[label]] = 1
                        
                        self.patches.append(patch)
                        self.patch_indices.append(((x_patch_min, y_patch_min), (x_patch_max, y_patch_max)))
                        """
                        if np.any((points[:, 0] >= i) & (points[:, 0] < i + self.patch_size[0]) & \
                                  (points[:, 1] >= j) & (points[:, 1] < j + self.patch_size[1])):
                            label_patch = torch.zeros(self.patch_size[0], self.patch_size[1], len(self.id_to_name))
                            label_patch[:, :, list(self.id_to_name.keys()).index(label)] = 1
                            labels.append(label_patch)
                            self.patches.append(patch)
                            self.patch_indices.append(((i, j), (i+self.patch_size[0], j+self.patch_size[1])))
                        """

def create_dataset(data_config, inflate_3d=True, transform=None, train=True):
    """Create a dataset based on the data_config
    Args:
        data_config (dict): Configuration dictionary containing the paths and parameters for the dataset.
        inflate_3d (bool): Whether to inflate the dataset to 3D. Default is True.
        transform (callable, optional): Optional transform to be applied on a sample.
        train (bool): Whether the dataset is for training or testing. Default is True.
    Returns:
        HyperspectralDataset: An instance of the dataset class.
    """
    paths_file = data_config['paths_file']
    
    with open(paths_file) as f:
        first_line = f.readline().strip()
        if first_line.endswith('.json'):
            dataset = PolygonDataset(data_config, inflate_3d=inflate_3d, transform=transform, train=train)
        else:
            dataset = NumpyDataset(data_config, inflate_3d=inflate_3d, transform=transform, train=train)
    
    return dataset
    