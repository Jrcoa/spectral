import torch
from src.modelsrc.unet3d import UNet3dResBlock, UNet3dResBlockV2, UNet3dResBlockV3
from src.modelsrc.unet3d import Net3dResBlock
from src.modelsrc.utils import get_latest_checkpoint_epoch


class ModelFactory:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.epoch_start = 0
        
    def build_model(self):
        if self.config["model"]["architecture"] == "unet3d":
            self.model = UNet3dResBlock(classes=self.config["data"]["classes"], bands=self.config['data']['in_bands'])
        elif self.config["model"]["architecture"] == "unet3d_v2":
            self.model = UNet3dResBlockV2(classes=self.config["data"]["classes"], dropout=True, bands=self.config['data']['in_bands'])
        elif self.config["model"]["architecture"] == "unet3d_v3":
            self.model = UNet3dResBlockV3(classes=self.config["data"]["classes"], dropout=True, bands=self.config['data']['in_bands'])
        elif self.config["model"]["architecture"] == "net3d":
            self.model = Net3dResBlock(classes=self.config["data"]["classes"], bands=self.config['data']['in_bands'])
        else:
            raise ValueError(f"Unsupported model architecture: {self.config['model']['architecture']}")

    def resume(self):
        # get the latest checkpoint
        checkpoint_epoch = get_latest_checkpoint_epoch(self.config["training"]["checkpoint_dir"], prefix="model_epoch_")
        checkpoint_file = f'model_epoch_{checkpoint_epoch}.pt'
        self.model.load_state_dict(torch.load(f'{self.config["training"]["checkpoint_dir"]}/{checkpoint_file}'))
        print(f"Resuming model from {checkpoint_file}")
        self.epoch_start = int(checkpoint_file.split('_')[-1].split('.')[0])    

