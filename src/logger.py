import os
from src.modelsrc.utils import get_latest_checkpoint_epoch
from torch import save

class Logger:
    def __init__(self, log_dir, checkpoint_dir):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        #os.makedirs(log_dir, exist_ok=True)
        #os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.map_file = f'{log_dir}/map_scores.txt'
        self.losses_file = f'{log_dir}/losses.txt'
        self.train_accuracy_file = f'{log_dir}/train_accuracy.txt'
        self.test_accuracy_file = f'{log_dir}/test_accuracy.txt'
        self.test_losses_files = f'{log_dir}/test_losses.txt'

        self.losses = []
        self.maps = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.test_losses = []
        
        self.files = [self.map_file, self.losses_file, self.train_accuracy_file, self.test_accuracy_file, self.test_losses_files]
        self.lists = [self.maps, self.losses, self.train_accuracy, self.test_accuracy, self.test_losses]
        
    def resume(self):
        checkpoint_epoch = get_latest_checkpoint_epoch(self.checkpoint_dir, prefix="model_epoch_")
        
        for file, lst in zip(self.files, self.lists):
            with open(file, 'r') as f:
                for line in f:
                    if line.startswith("Epoch"):
                        epoch_val = int(line.split(" ")[1])
                        if epoch_val <= checkpoint_epoch:
                            lst.append(float(line.split(":")[-1].strip()))
    
    def update_file(self, name, value, epoch):
        path = os.path.join(self.log_dir, f"{name}.txt")
        with open(path, "a") as f:
            f.write(f"Epoch {epoch} {name.replace('_', ' ').title()}: {value:.4f}\n")
        
    def update_train_accuracy(self, accuracy):
        self.train_accuracy.append(accuracy)
        self.update_file("train_accuracy", accuracy, len(self.train_accuracy))

    def update_test_accuracy(self, accuracy):   
        self.test_accuracy.append(accuracy)
        self.update_file("test_accuracy", accuracy, len(self.test_accuracy))
        
    def update_losses(self, loss):  
        self.losses.append(loss)
        self.update_file("losses", loss, len(self.losses))
        
    def update_test_losses(self, loss): 
        self.test_losses.append(loss)
        self.update_file("test_losses", loss, len(self.test_losses))
        
    def update_maps(self, map_score):
        self.maps.append(map_score)
        self.update_file("map_scores", map_score, len(self.maps))

    
    def update_best_model(self, model, map_score, accuracy):
        if len(self.maps) == 0 or map_score > max(self.maps):
            save(model.state_dict(), f'{self.checkpoint_dir}/model_best_map.pt')
        if len(self.test_accuracy) == 0 or accuracy > max(self.test_accuracy):
            save(model.state_dict(), f'{self.checkpoint_dir}/model_best_accuracy.pt')