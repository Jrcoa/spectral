import argparse
import os
from osgeo import gdal
import spectral.io.envi as envi

def convert_to_tif(hdr_file):
    img = envi.open(hdr_file)
    data = img.load().asarray()
    
    driver = gdal.GetDriverByName('GTiff')
    out_file = hdr_file.replace('.hdr', '.tif')
    out_ds = driver.Create(out_file, img.ncols, img.nrows, img.nbands, gdal.GDT_Float32)
    
    for i in range(img.nbands):
        out_ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])
    
    out_ds.FlushCache()
    print(f"Converted {hdr_file} to {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert ENVI HDR hyperspectral files to TIFF format.')
    parser.add_argument('input_directory', type=str, help='The directory containing the HDR files to convert')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_directory):
        print(f"The directory {args.input_directory} does not exist.")
        return
    
    # Process the files in the directory
    for filename in os.listdir(args.input_directory):
        file_path = os.path.join(args.input_directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.hdr'):
            print(f"Processing file: {file_path}")
            convert_to_tif(file_path)

if __name__ == "__main__":
    main()
