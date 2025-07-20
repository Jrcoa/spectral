import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import argparse
from tifffile import imread
import pandas as pd
from src.dataloader import HyperspectralDataset
from src.utils import latlong_to_pixel
from src.utils import load_image
from src.utils import parse_points_file
import cv2

def plot_ground_surveys(img, points, names):
    unique_names = list(set(names))
    colors = plt.cm.get_cmap('nipy_spectral', len(unique_names))
    colors = {name: colors(i) for i, name in enumerate(unique_names)}

    fig, ax = plt.subplots()
    ax.imshow(img.astype(np.uint8))
    for (x, y), name in zip(points, names):
        circ = plt.Circle((x, y), radius=10, color=colors[name], fill=True)
        ax.add_patch(circ)
        ax.text(x + 12, y, name[:5] + "..." + name[-3:], fontsize=6)  # Add short class name   
    ax.legend([plt.Circle((0, 0), radius=1, color=colors[name], fill=True) for name in unique_names], unique_names, loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize='x-small')
    #plt.title(f"{args.image_path.split('\\')[-1].split('/')[-1]}")
    plt.axis('off')
    plt.show()
    
def plot_patch_class_distribution(args):
    dataloader = HyperspectralDataset(args.image_path, args.metadata_file, patch_size=args.patch_size, stride=args.stride)
    
    class_patch_counts = {name : 0 for name in dataloader.id_to_name.values()} # how many patches contain the class
    class_pixel_counts = {name : 0 for name in dataloader.id_to_name.values()} # how many pixels in total for the class across patches
    
    for i, (image, label) in enumerate(dataloader):
        # label is shape <classes, patch_size[0], patch_size[1]>
        # image is shape <channels, patch_size[0], patch_size[1]>
        label = label.numpy().argmax(axis=0)
        for name_id in dataloader.id_to_name.keys():
            name = dataloader.id_to_name[name_id]
            print(name, name_id)
            if np.any(label == name_id):
                class_patch_counts[name] += 1
                class_pixel_counts[name] += np.sum(label == name_id)
    
    hist = pd.DataFrame({'class': list(class_patch_counts.keys()), 'patch_count': list(class_patch_counts.values()), 'pixel_count': list(class_pixel_counts.values())})
    hist = hist.sort_values(by='patch_count', ascending=False)
    print(hist)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    width = 0.4  # width of the bars
    x = np.arange(len(hist['class']))  # the label locations
    ax1.bar(x, hist['patch_count'], width, label='Patch Count')
    ax2.bar(x, hist['pixel_count'], width, label='Pixel Count')
    # get rid of ax1 ticks
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Pixel Count')
    ax1.set_ylabel("Patch Count")
    ax1.set_title('Patch and Pixel Count by Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(hist['class'], rotation=45, ha='right')
    #ax.legend()

    plt.tight_layout()

    plt.show()

def main(args):
    
    if args.image_path:
        file_path = args.image_path
        i, map_info = load_image(file_path)
        print(f"Loaded file shape: {i.shape}")
        print(f"Maximum value: {i.max()}")
        print(f"Minimum value: {i.min()}")
        
        if len(i.shape) == 3:
            # normalize bands individually
            i_norm = i
            i_norm = (i_norm - i_norm.min(axis=(0, 1), keepdims=True)) / \
                    (i_norm.max(axis=(0, 1), keepdims=True) - i_norm.min(axis=(0, 1), keepdims=True))
            pca = PCA(n_components=3)        
            flat_data = i_norm.reshape(-1, i_norm.shape[-1])
            img = pca.fit_transform(flat_data)
            img[np.abs(img) < 9e-2] = 0
        else:
            img = i.copy()

        # normalize pca components and reshape
        #img = (img - img.min()) / (img.max() - img.min()) 
        img = (img - img.min(axis=0)) / img.max(axis=0)
        img = img.reshape(i.shape[0], i.shape[1], -1) * 255
        # save img as png, not using matplotlib
        img2 = img.astype(np.uint8)
        img2 = np.clip(img2, 0, 255)
        #img2 = np.transpose(img2, (1, 0, 2))  # transpose to (height, width, channels)
        #img2 = img2[:, :, ::-1]  # convert from RGB to BGR
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)  
        cv2.imwrite('out.png', img2)  # save as png
        if args.points_file:
            print(map_info)
            points, names = parse_points_file(args.points_file, map_info)
            plot_ground_surveys(img, points, names)
            for name, (x, y) in zip(names, points):
                print(f"{name}: ({x}, {y})")
        else:        
            plt.imshow(img.astype(np.uint8))
            plt.title(f"{args.image_path.split('\\')[-1].split('/')[-1]}")
            plt.axis('off')
            plt.show()
    else:
        plot_patch_class_distribution(args)

class TupleStoreAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(map(int, values.split(','))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an HDR or TIF file and visualize it using PCA.")
    parser.add_argument("--image_path", type=str, help="Path to the HDR or TIF file to process.")
    parser.add_argument("--points_file", required=False, default='', type=str, help="Path to the CSV or XLSX file containing latitude and longitude points.")
    
    parser.add_argument("--paths_file", type=str, help="Path to the paths file.")
    parser.add_argument("--metadata_file", type=str, help="Path to the metadata file.")
    parser.add_argument("--patch_size", type=str, default=(128,128), help="Size of the patches \"sx, sy\" .", action=TupleStoreAction)
    parser.add_argument("--stride", type=int, default=128, help="Stride of the patches.")
    
    
    args = parser.parse_args()

    main(args)
