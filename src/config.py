import os
import yaml
import torch
from typing import Dict, Any

class Config(Dict[str, Any]):
    def __init__(self, path: str):
        self.required_keys = [
        "experiment", "data", "model", "training", "seed", "device"
        ]
        self.data_keys = ["paths_file", "metadata_file", "patch_size", "stride", "classes", "transforms", "shuffle"]
        self.training_keys = ["optimizer", "learning_rate", "weight_decay", "epochs", "checkpoint_dir", "log_dir", "log_interval", "checkpoint_interval", "batch_size", "grad_clip"]
    
        self.config = self.load_config(path)
        self.validate()
        
        super().__init__(self.config)

    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def validate(self):
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")

        # Verify experiment section
        if "name" not in self.config["experiment"] or "description" not in self.config["experiment"]:
            raise ValueError("Missing 'name' or 'description' in 'experiment' section.")

        # Verify data section
        for key in self.data_keys:
            if key not in self.config["data"]:
                raise ValueError(f"Missing '{key}' in 'data' section.")

        if not "in_bands" in self.config["data"]:
            print("in_bands not specified, using all bands")
            self.config["data"]["in_bands"] = -1 # use all bands

        # If not using PCA, set the whiten parameter to False
        if self.config["data"]["in_bands"] == -1:
            self.config["data"]["whiten"] = False
        else:
            assert "whiten" in self.config["data"], "whiten parameter must be set to True or False if in_bands is specified"
        
        # Verify model section
        if "architecture" not in self.config["model"]:
            raise ValueError("Missing 'architecture' in 'model' section.")

        # Verify training section
        for key in self.training_keys:
            if key not in self.config["training"]:
                raise ValueError(f"Missing '{key}' in 'training' section.")
        self.config['training']['weight_decay'] = float(self.config['training']['weight_decay'])
        self.config['training']['learning_rate'] = float(self.config['training']['learning_rate'])
        self.config['training']['grad_clip'] = float(self.config['training']['grad_clip'])
        
        if not "scheduler" in self.config["training"]:
            self.config["training"]["scheduler"] = None
        elif not "name" in self.config["training"]["scheduler"]:
            raise ValueError("Missing 'name' in 'scheduler' section.")
        
        if self.config["training"]["scheduler"] is not None and self.config["training"]["scheduler"]["name"] == "cosinewarmrest":
            scheduler_keys = ["T_0", "T_mult", "eta_min"]
            for key in scheduler_keys:
                if key not in self.config["training"]["scheduler"]:
                    raise ValueError(f"Missing '{key}' in 'scheduler' section.")   
            self.config["training"]["scheduler"]["T_0"] = int(self.config["training"]["scheduler"]["T_0"])
            self.config["training"]["scheduler"]["T_mult"] = int(self.config["training"]["scheduler"]["T_mult"])
            self.config["training"]["scheduler"]["eta_min"] = float(self.config["training"]["scheduler"]["eta_min"])
            
        elif self.config["training"]["scheduler"] is not None and self.config["training"]["scheduler"]["name"] == "cosine":
            scheduler_keys = ["eta_min"]
            for key in scheduler_keys:
                if key not in self.config["training"]["scheduler"]:
                    raise ValueError(f"Missing '{key}' in 'scheduler' section.")
            self.config["training"]["scheduler"]["eta_min"] = float(self.config["training"]["scheduler"]["eta_min"])
            
        elif self.config["training"]["scheduler"] is not None and self.config["training"]["scheduler"]["name"] == "step":
            scheduler_keys = ["step_size", "gamma"]
            for key in scheduler_keys:
                if key not in self.config["training"]["scheduler"]:
                    raise ValueError(f"Missing '{key}' in 'scheduler' section.")
            self.config["training"]["scheduler"]["step_size"] = int(self.config["training"]["scheduler"]["step_size"])
            self.config["training"]["scheduler"]["gamma"] = float(self.config["training"]["scheduler"]["gamma"])
        
        if not "visualize_every" in self.config["training"]:
            self.config["training"]["visualize_every"] = 0 # disable

        # Verify seed and device
        if not isinstance(self.config["seed"], int):
            raise ValueError("'seed' must be an integer.")
        if self.config["device"] not in ["cuda", "cpu"]:
            raise ValueError("'device' must be either 'cuda' or 'cpu'.")

        # make sure the specific cuda device (eg cuda:1) is available
        if self.config["device"] == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")
        elif "cuda" in self.config["device"]:
            if ":" not in self.config["device"]:
                raise ValueError("Device ID must be specified in the format 'cuda:<device_id>'.")
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")
            device_id = int(self.config["device"].split(":")[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"Device ID {device_id} is not available.")
        elif self.config["device"] != "cpu":
            raise ValueError(f"Unsupported device: {self.config['device']}")
        
        # verify that paths exist
        if not os.path.exists(self.config["training"]["checkpoint_dir"]):
            raise ValueError(f"Checkpoint directory {self.config['training']['checkpoint_dir']} does not exist.")
        if not os.path.exists(self.config["training"]["log_dir"]):
            raise ValueError(f"Log directory {self.config['training']['log_dir']} does not exist.")
        if not os.path.exists(self.config["data"]["paths_file"]):
            raise ValueError(f"Paths file {self.config['data']['paths_file']} does not exist.")
        if not os.path.exists(self.config["data"]["metadata_file"]):
            raise ValueError(f"Metadata file {self.config['data']['metadata_file']} does not exist.")
