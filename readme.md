# Spectral

Research code and library for training deep neural networks on hyperspectral imagery

Author: Joseph Casale

## Features

- Tools for constructing a spectral_dnn formatted dataset from hdr files
- Visualizations tools using PCA combining HSI and ground survey data
- Training script configurable via yaml file
- Compatible dataset formats: custom spectral_dnn format and labelme polygon json format, format automatically detected

TO BE UPDATED:
- map score calculations
- lidar/hsi fusion
- processing full directories of images is theoretically possible with minor
modifications to dataloader.py -- currently hard-coded to throw exception in this case. Delete these lines mindfully. 

## Installation

```bash
git clone https://github.com/jrcoa/spectral.git
cd spectral
# Follow setup instructions for your environment
```

## Usage
(To be expanded)
entry points are construct_dataset.py, train.py, visualize.py

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

