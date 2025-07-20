import argparse

from matplotlib import pyplot as plt
import numpy as np
from src.utils import load_image, parse_points_file
from collections import deque
import torch 

def build_target_matrix(hdr_file, points_file):
    """Builds a target matrix from the given HDR image file and points file.
        hdr_file (str): Path to the HDR image file.
        points_file (str): Path to the points file containing coordinates and labels.
        tuple: A tuple containing:
            - canvas (np.ndarray): A 2D array with the same dimensions as the image, where each point is labeled with an integer ID.
            - labeled_points (list): A list of tuples (x, y, id) where id corresponding label ID to point (x,y).
    """
    image, map_info = load_image(hdr_file)
    points, names = parse_points_file(points_file, map_info)
    
    names_to_ids = {n : i for i, n in enumerate(list(set(names)))}
    labeled_points = [(x, y, names_to_ids[name], name) for (x, y), name in zip(points, names)]
    
    image_dimensions = image.shape[:2]
    canvas = np.zeros(image_dimensions)
    for x, y, id, name in labeled_points:
        try:
            canvas[y, x] = id
        except IndexError:
            print(f"Point ({x}, {y}) is outside the image bounds")
    return canvas, labeled_points

def interpolate_target_matrix_slow(target_matrix, points):
    # we have to assign every pixel to the nearest tree
    for i in range(target_matrix.shape[0]):
        for j in range(target_matrix.shape[1]):
            if target_matrix[i, j] == 0:
                min_distance = float('inf')
                # for every point, calculate the distance to the pixel
                for (x, y, id, _) in points:
                    distance = (i - y)**2 + (j - x)**2
                    if distance < min_distance:
                        min_distance = distance
                        target_matrix[i, j] = id
    return target_matrix


def save_target_image(hdr_file, points_file, output_file, x_start=0, x_end=None, y_start=0, y_end=None):
    target_array, labeled_points = build_target_matrix(hdr_file, points_file)
    target_array = interpolate_target_matrix_slow(target_array, labeled_points)
    # Increase the amount of the array printed
    np.set_printoptions(threshold=np.inf)
    print(target_array)
    # save as image file
    target_image = np.stack([target_array, target_array, target_array], axis=-1)
    target_image = target_image.astype(np.uint8)
    plt.imsave(output_file + '_pre-cropped.png', target_image)
    print(f"Saved pre-cropped target image to {output_file}_pre-cropped.png")
    
    # convert the target array to one hots
    n_classes = len(set(target_array.flatten()))
    print("Number of classes: ", n_classes)
    target_array = torch.tensor(target_array)
    target_array = target_array.long()
    target_array = torch.nn.functional.one_hot(target_array, num_classes=n_classes)
    #target_array = target_array.permute(2, 0, 1).float() # <num_classes>, <height>, <width>
    target_array = target_array.float() # <height>, <width>, <num_classes>
    
    x_start = int(x_start * target_image.shape[1])
    y_start = int(y_start * target_image.shape[0])
    if x_end is not None:
        x_end = int(x_end * target_image.shape[1])
    if y_end is not None:
        y_end = int(y_end * target_image.shape[0])
    
    target_array = target_array[y_start:y_end, x_start:x_end, :]
    target_image = target_image[y_start:y_end, x_start:x_end]
    torch.save(target_array, output_file + '_cropped.pt')
    plt.imsave(output_file + '_cropped.png', target_image)
    print(f"Saved cropped target image to {output_file}_cropped.png")
    print(f"Saved cropped target tensor to {output_file}_cropped.pt")
    
    # Saving a map of id to names and crop information and size of the image
    id_to_name = {id : name for _, _, id, name in labeled_points}
    with open(output_file + '_meta.txt', 'w') as f:
        for id, name in id_to_name.items():
            f.write(f"pair:{id},{name}\n")
        f.write(f"crop:{x_start},{x_end},{y_start},{y_end}\n")
        f.write(f"size:{target_image.shape[0]},{target_image.shape[1]}\n") # height, width
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert tree points file paired with an hdr file to a txt file containing tree pixel indices")
    parser.add_argument("hdr_file", type=str, help="Path to the HDR file to process.")
    parser.add_argument("points_file", type=str, help="Path to the CSV or XLSX file containing latitude and longitude points.")
    parser.add_argument("output_file", type=str, help="Path to the output file.")
    
    parser.add_argument("--x_start", type=float, default=0, help="X start (as percentage of image width)")
    parser.add_argument("--x_end", type=float, default=None, help="X end (as percentage of image width)")
    parser.add_argument("--y_start", type=float, default=0, help="Y start (as percentage of image height)")
    parser.add_argument("--y_end", type=float, default=None, help="Y end (as percentage of image height)")

    args = parser.parse_args()
    
    save_target_image(args.hdr_file, args.points_file, args.output_file, args.x_start, args.x_end, args.y_start, args.y_end)
    
    