import math
from cv2 import imread
import pandas as pd
from sklearn.decomposition import PCA
import spectral as sp
import torch

# WGS 1984 Parameters
a = 6378137
f = 1 / 298.257223563
n = f / (2 - f)
A = a / (1 + n) * (1 + n**2/4 + n**4/64 + n**6/256 + n**8/16384)
k0 = 0.9996
alpha1 = n/2 - 2*n**2/3 + 37*n**3/96 - n**4/360
alpha2 = n**2 * 13/48 - 3/5 * n**3
alpha3 = 61/240 * n**3

def train_pca(img, n_components=3):
    i_norm = img.copy()
    i_norm = (img - img.min(axis=(0, 1), keepdims=True)) / \
            (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
    pca = PCA(n_components=n_components)
    pca.fit(i_norm.reshape(-1, img.shape[-1]))
    return pca     


def latlong_to_utm(lat, lon):
    """Convert latitude and longitude to UTM coordinates (WGS 1984).
    https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    Assumes zone 1 for the southern hemisphere.
    """
    
    zone = math.floor((lon + 180) / 6) + 1
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    lon0 = (zone - 1) * 6 - 180 + 3 
    lon0_rad = math.radians(lon0)

    t = math.sinh(math.atanh(math.sin(lat_rad)) - (2 * math.sqrt(n) / (1 + n)) * 
                  math.atanh((2 * math.sqrt(n) / (1 + n)) * math.sin(lat_rad)))

    epsilon = math.atan(t / math.cos(lon_rad - lon0_rad))
    nu = math.atanh(math.sin(lon_rad - lon0_rad) / math.sqrt(1 + t**2))
    
    E = k0 * A * (nu + alpha1 * math.cos(2 * epsilon) * math.sinh(2 * nu) +
                  alpha2 * math.cos(4 * epsilon) * math.sinh(4 * nu) +
                  alpha3 * math.cos(6 * epsilon) * math.sinh(6 * nu))
    
    N = k0 * A * (epsilon + alpha1 * math.sin(2 * epsilon) * math.cosh(2 * nu) +
                  alpha2 * math.sin(4 * epsilon) * math.cosh(4 * nu) +
                  alpha3 * math.sin(6 * epsilon) * math.cosh(6 * nu))

    # Have to add "false" easting and northing
    E = E + 500000
    if lat < 0:
        N = N + 10000000

    return E, N, zone

def latlong_to_pixel(lat, lon, ref_x, ref_y, coord_x, coord_y, pixel_size_x, pixel_size_y, x_start=0, y_start=0, **kwargs):
    """Convert latitude and longitude to pixel coordinates in image.
    
    Parameters:
    lat, lon     -> Latitude and longitude of the point.
    ref_x, ref_y -> Reference pixel coordinates (ref_x, ref_y in map info).
    coord_x, coord_y -> UTM coordinates of reference pixel (coord_x, coord_y in map info).
    pixel_size_x, pixel_size_y -> Pixel resolution (in meters).
    x_start, y_start -> Image start indices (default: 0,0 for full image).

    Returns:
    (px, py) -> Pixel coordinates (column, row).
    """
    E, N, _ = latlong_to_utm(lat, lon)
    # E and N are measured in meters so we can convert to pixels
    pixel_x = (E - (coord_x)) / pixel_size_x + (ref_x-1) 
    pixel_y = ((coord_y) - N) / pixel_size_y + (ref_y-1) 
    
    pixel_x = int(round(pixel_x))
    pixel_y = int(round(pixel_y))
    
    if pixel_x < 0 or pixel_y < 0:
        return 0, 0
    return pixel_x, pixel_y

def __convert_to_float(value):
    try:
        return float(value)  
    except ValueError:
        return value 
     
def load_image(file_path):
    map_info = {}
    if file_path.endswith('.hdr'):
        img = sp.envi.open(file_path).asarray().copy()
        with open(file_path) as f:
            for line in f:
                if line.startswith("map info"):
                    map_str = line.split("{")[1].split("}")[0]
                    names = ['ref_x', 'ref_y', 'coord_x', 'coord_y', 'pixel_size_x', 'pixel_size_y', 'zone']
                    values = [s.strip() for s in map_str.strip().split(',')[1:8]]
                    map_info.update({names[i] : __convert_to_float(values[i]) for i in range(len(values))})
                elif line.startswith('x start'):
                    map_str = line.split("=")[1].strip()
                    map_info['x_start'] = int(map_str)
                elif line.startswith('y start'):
                    map_str = line.split("=")[1].strip()
                    map_info['y_start'] = int(map_str)
        

    elif file_path.endswith('.tif') or file_path.endswith('.tiff'):
        img = imread(file_path)
    else:
        raise ValueError("Unsupported file format")
    return img, map_info

def parse_points_file(points_file, data_dict, name_type="Local Names"):
    """_summary_

    Args:
        points_file (_type_): _description_
        data_dict (_type_): _description_
        name_type (str, optional): _description_. Defaults to "Local Names". This is recommended, "Scientific Name" has nans.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if points_file.endswith('.csv'):
        df = pd.read_csv(points_file)
    elif points_file.endswith('.xlsx'):
        df = pd.read_excel(points_file,  header=1)
    else:
        raise ValueError("Unsupported file format")
    points = []
    names = []
    for _, row in df.iterrows():
        lat, lon, name = row['Latitude'], row['Longitude'], row[name_type]
        x, y = latlong_to_pixel(float(lat), float(lon), **data_dict)
        points.append((x, y))
        names.append(name)
    return points, names

def dice_loss(pred, target):
    """
    Computes the Dice Loss, a measure of overlap between two samples.
    Args:
        preds (torch.Tensor): The predicted output tensor <batch_size> , <H>, <W>, <classes> softmaxed
        target (torch.Tensor): The ground truth tensor <batch_size> , <H>, <W>, <classes>
    Returns:
        float: The computed Dice Loss, where a lower value indicates better overlap.
    """
    #pred = torch.nn.functional.softmax(pred, dim=3) 
    num_classes = pred.shape[3]  
    dice = 0  
    eps = 1e-6
    
    for c in range(num_classes):  
        pred_c = pred[:, :, :, c]  # Predictions for class c
        target_c = target[:, :, :, c]  # Ground truth for class c
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        
        dice += (2. * intersection + eps) / (union + eps)  # Per-class Dice score

    return 1 - dice.mean() / num_classes  # Average Dice Loss across classes

def TV(pred):
    """_summary_

    Args:
        preds (torch.Tensor): The predicted output tensor <batch_size> , <H>, <W>, <classes>
        target (torch.Tensor): The ground truth tensor <batch_size> , <H>, <W>, <classes>
    Returns:
        _type_: _description_
    """
    #pred_soft = torch.functional.softmax(pred, dim=3)
    dh = torch.abs(pred[:, 1:, :, :] - pred[:, :-1, :, :])  # vertical diff
    dw = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])  # horizontal diff
    return (dh.sum() + dw.sum())
    