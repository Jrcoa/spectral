import torch
import torch.nn as nn
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from src.utils import dice_loss, TV
from src.logger import Logger
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.device = None
        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.dataloader = None
        self.test_dataloader = None
        self.dataset = None
        self.test_dataset = None
        self.criterion = None
        self.transforms = None
        self.targ_bins = None       

    def setup(self, model_factory, logger, config, optimizer, scheduler, criterion, train_data, test_data = None):
                
        torch.cuda.empty_cache() 
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        self.device = torch.device(config["device"])
        self.model : nn.Module = model_factory.model.to(self.device)
        self.config = config
        self.logger : Logger = logger
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(self.device)
        self.targ_bins = [0 for _ in range(config["data"]["classes"])]        

        self.dataloader = DataLoader(self.train_data, batch_size=config["training"]["batch_size"], shuffle=True)
        self.epoch_start = model_factory.epoch_start

        if self.test_data is not None:
            self.test_dataloader = DataLoader(self.test_data, batch_size=config["training"]["batch_size"], shuffle=False)
        else:
            self.test_dataloader = None
            print("Warning: test_data parameter not provided, skipping testing.")
        
    def train_epoch(self, epoch):
        visualize = self.config["training"]["visualize_every"] > 0 and epoch % self.config["training"]["visualize_every"] == 0
              
        preds_bins = [0 for _ in range(self.config["data"]["classes"])]
        epoch += self.epoch_start
        self.model.train()
        
        epoch_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_tv = 0.0
        data = tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch}")
 
        accuracy = 0
        correct = 0 
        total = 0
        
        self.optimizer.zero_grad()
        loss = 0.0  # Accumulate loss

        for i, (image, label) in data:
            # label: <batch, H, W, classes>
            # image = <batch>, <channels>, <depth>, <height>, <width>
            image, label = image.to(self.device), label.to(self.device) 
            with torch.amp.autocast("cuda"): 
                output = self.model(image)  # output: (batch, H, W, classes)
                preds = torch.nn.functional.softmax(output, dim=3)
                batch_loss_dice = dice_loss(preds, label)
                tv = TV(preds)
                
                output = output.reshape(output.shape[0] * output.shape[1] * output.shape[2], output.shape[3]) # <batch * num_pixels>, <classes>
                label = label.reshape(label.shape[0] * label.shape[1] * label.shape[2], label.shape[3]) # <batch * num_pixels>, <classes>s
                label = label.argmax(dim=1) # <batch * num_pixels>
                batch_loss_ce = self.criterion(output, label)  
                batch_loss = 1 * batch_loss_ce + 2 * batch_loss_dice + 0.001 * tv
            
            if batch_loss.isnan().any():
                print("Loss is NaN, terminating...")
                return
            
            # Scale loss and backpropagate, update optimizer
            (batch_loss / self.accumulation_steps).backward()
            if (i + 1) % self.accumulation_steps == 0 or i == len(self.dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
                self.optimizer.step()
                self.optimizer.zero_grad()   
                data.set_description(f"Epoch {epoch} loss: {epoch_loss / (i+1):.4f} Dice Loss: {epoch_dice_loss / (i+1):.4f} TV: {epoch_tv / (i+1):.4f} CE Loss: {epoch_ce_loss / (i+1):.4f}")
            
            # Update loss metrics
            epoch_loss += batch_loss.item() / self.accumulation_steps
            epoch_dice_loss += 2 * batch_loss_dice.item() / self.accumulation_steps
            epoch_ce_loss += 1 * batch_loss_ce.item() / self.accumulation_steps
            epoch_tv += 0.0003 *tv.item() / self.accumulation_steps
                
                
            # Accumulate accuracy metrics for this epoch
            with torch.no_grad():
                
                preds = preds.reshape(-1, preds.shape[3]).argmax(dim=1) # <batch * H * W>, <classes>
                correct += (preds == label).sum().item()
                total += label.size(0)
                
                if i == 0 and visualize:
                    num_rows = self.config["training"]["batch_size"] if self.config["training"]["batch_size"] < 5 else 5
                    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 20))
                    fig.subplots_adjust(hspace=0.5, wspace=0.5)
                    axs = axs.flatten()
            
                    label_reshaped = label.reshape(image.shape[0], image.shape[3], image.shape[4])
                    preds_reshape = preds.reshape(image.shape[0], image.shape[3], image.shape[4])
                    
                    # pca all of the patches and plot them along with labels and predictions                                     
                    for j in range(num_rows):
                        label_patch_img = label_reshaped[j].cpu().numpy() * 255 # <height, width>
                        label_patch_img = label_patch_img.astype(np.uint8)
                        label_patch_img = np.stack([label_patch_img, label_patch_img, label_patch_img], axis=-1) # <height, width, 3>
                        
                        patch_pca = self.train_data.pca_transform_patch(image[j]) # output is <3, H, W> numpy array
                        patch_pca = patch_pca.transpose(1, 2, 0) * 255 # <H, W, 3>
                        patch_pca = patch_pca.astype(np.uint8)
                        
                        pred_patch_img = preds_reshape[j].cpu().numpy() * 255 # <height, width>
                        pred_patch_img = pred_patch_img.astype(np.uint8)
                        pred_patch_img = np.stack([pred_patch_img, pred_patch_img, pred_patch_img], axis=-1) # <height, width, 3>
                        
                        axs[j * 3].imshow(label_patch_img)
                        axs[j * 3].set_title(f"Label Patch {j}")
                        axs[j * 3].axis('off')
                        axs[j * 3 + 2].imshow(patch_pca)
                        axs[j * 3 + 2].set_title(f"Patch PCA {j}")
                        axs[j * 3 + 2].axis('off')
                        axs[j * 3 + 1].imshow(pred_patch_img)
                        axs[j * 3 + 1].set_title(f"Pred Patch {j}")
                        axs[j * 3 + 1].axis('off')
                    plt.savefig(f'{self.config["training"]["log_dir"]}/epoch_{epoch}_train_batch_{i}.png')
                    plt.close()
                
                # Update class distribution for target if first batch
                # and for predictions if log interval
                if epoch == 0:
                    for i in range(len(label)):
                        self.targ_bins[label[i]] += 1
                if epoch % self.config["training"]["log_interval"] == 0:    
                    for i in range(len(preds)):
                        preds_bins[preds[i]] += 1
        
        # End of epoch 
        self.scheduler.step()
        accuracy = correct/total
        #train_env.update_train_accuracy(accuracy)
        
        print(f"Epoch {epoch} Train Accuracy: {accuracy:.4f}")
        # Plot class distributions with overlap colored differently
        if epoch % (self.config["training"]["log_interval"]) == 0:
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.config["data"]["classes"]), self.targ_bins, alpha=0.5, label='Target', color='blue')
            plt.bar(range(self.config["data"]["classes"]), preds_bins, alpha=0.5, label='Predicted', color='orange')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.legend()
            plt.savefig(f'{self.config["training"]["log_dir"]}/class_distribution_epoch_{epoch}.png')
            plt.close()
        
        #train_env.update_losses(epoch_loss / len(train_env.dataloader))
        if epoch % self.config["training"]["checkpoint_interval"] == 0 or epoch == self.config["training"]["epochs"] - 1:
            torch.save(self.model.state_dict(), f'{self.config["training"]["checkpoint_dir"]}/model_epoch_{epoch}.pt')
        
        self.logger.update_train_accuracy(accuracy)
        self.logger.update_losses(epoch_loss / len(self.dataloader))
        
        return accuracy, epoch_loss  

    def test_epoch(self, epoch):
        if self.test_dataloader is None:
            print("No test data provided, skipping testing.")
            return 0, 0, 0
        
        self.model.eval()
        test_preds = None
        test_labels = None
        
        with torch.no_grad():
            test_loss = 0.0
            test_data = tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), desc=f"Epoch {epoch}")

            for i, (image, label) in test_data:
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                output = output.reshape(output.shape[0] * output.shape[1] * output.shape[2], output.shape[3]) # <batch * num_pixels>, <classes>
                label = label.reshape(label.shape[0] * label.shape[1] * label.shape[2], label.shape[3]).argmax(1) # <batch * num_pixels>, <classes>
                loss = self.criterion(output, label)
                test_loss += loss.item()
                
                #preds = nn.functional.softmax(output.reshape(-1, output.shape[2]), dim=1).cpu().numpy()
                label = label.cpu().numpy()
                preds = nn.functional.softmax(output, dim=1).argmax(dim=1).cpu().numpy()
                test_preds = np.append(test_preds, preds, axis=0) if test_preds is not None else preds
                test_labels = np.append(test_labels, label, axis=0) if test_labels is not None else label
               
            # Metric Computations 
            #map_score = average_precision_score(test_labels.T, test_preds.T, average='weighted')
            map_score = 0 # too slow, implement later
            #test_labels = test_labels.argmax(axis=1)
            #test_preds = test_preds.argmax(axis=1)
            correct = (test_preds == test_labels).sum()
            total = test_labels.size
            accuracy = correct / total
            
            # Update best models
            self.logger.update_best_model(self.model, map_score, accuracy)
                
            # Save metrics
            self.logger.update_test_accuracy(accuracy)
            self.logger.update_test_losses(test_loss / len(self.test_dataloader))
            self.logger.update_maps(map_score)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Epoch {epoch} mAP Score: {map_score:.4f}")

            test_data.set_description(f"Epoch {epoch} Test Loss: {test_loss / len(self.test_dataloader):.4f}")
            
        return accuracy, map_score, test_loss / len(self.test_dataloader)