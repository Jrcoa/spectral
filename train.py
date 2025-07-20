ACCUMULATION_STEPS = 4

from src.dataloader import create_dataset
from src.config import Config
from src.transforms import RandomRotate, RandomHorizontalFlip, RandomVerticalFlip, Compose
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from src.modelsrc import ModelFactory
from src.logger import Logger
from src.trainer import Trainer


class DummyScheduler:
    def __init__(self): pass
    def step(self): pass
        
    
def train_model(model_config):
    config = Config(model_config)
    model_factory = ModelFactory(config)
    model_factory.build_model()
    logger = Logger(config["training"]["log_dir"], config["training"]["checkpoint_dir"])

    if config["model"]["resume"]:
        model_factory.resume()
        logger.resume()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None
    if config["training"]["optimizer"] == "adam":
        optimizer = optim.Adam(model_factory.model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    elif config["training"]["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(model_factory.model.parameters(), lr=config["training"]["learning_rate"], momentum=0.99, weight_decay=config["training"]["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")

    # Ensure 'initial_lr' is set for each parameter group when resuming
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = config["training"]["learning_rate"]
    
    if config["training"]["scheduler"]["name"] == "cosinewarmrest":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["training"]["scheduler"]["T_0"], 
                                                                        T_mult=config["training"]["scheduler"]["T_mult"],  
                                                                        eta_min=config["training"]["scheduler"]["eta_min"], 
                                                                        last_epoch=model_factory.epoch_start-1)
    elif config["training"]["scheduler"]["name"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=config["training"]["epochs"], 
                                                            eta_min=config["training"]["scheduler"]["eta_min"], 
                                                            last_epoch=model_factory.epoch_start-1)
    elif config["training"]["scheduler"]["name"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=config["training"]["scheduler"]["step_size"], 
                                                    gamma=config["training"]["scheduler"]["gamma"], 
                                                    last_epoch=model_factory.epoch_start-1)
    else:
        scheduler = DummyScheduler() # no scheduler

    if config["data"]["transforms"]:
        print("Enabling transformations...")
        transforms = Compose([RandomRotate(), RandomHorizontalFlip(), RandomVerticalFlip()])
    
    dataset = create_dataset(data_config=config["data"], transform=transforms, train=True)
    test_dataset = create_dataset(data_config=config["data"], transform=None, train=False)
    dataset.build()
    test_dataset.build()
    
    trainer = Trainer(accumulation_steps=ACCUMULATION_STEPS,)
    trainer.setup(model_factory, logger, config, optimizer, scheduler, criterion, train_data=dataset, test_data=test_dataset)
    print(f"Training model {config['model']['architecture']} with {config['data']['classes']} classes on {config['device']} device")
    
    
    print(f"Using {trainer.scheduler.__class__.__name__} scheduler")
    
    for epoch in range(model_factory.epoch_start, config["training"]["epochs"]):  
        
        accuracy, epoch_loss = trainer.train_epoch(epoch=epoch)
        
        # Test the model
        test_accuracy, map, test_loss = trainer.test_epoch(epoch=epoch)


        if epoch % config["training"]["log_interval"] == 0:
            plt.plot(logger.losses, label="Train Loss")
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss')
            #plt.legend()
            plt.savefig(f'{config["training"]["log_dir"]}/loss.png')
            plt.close()
            
            plt.plot(logger.test_losses, label="Test Loss")
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Test Loss')
            #plt.legend()
            plt.savefig(f'{config["training"]["log_dir"]}/test_loss.png')
            plt.close()
            
            last_losses = logger.losses[-30:]
            last_test_losses = logger.test_losses[-30:]
            plt.plot(last_losses, label="Train Loss")
            plt.plot(last_test_losses, label="Test Loss")
            plt.xlabel('Iteration')
            #plt.xlim((len(losses)-30, len(losses)))
            plt.xticks(list(range(len(last_losses))), [i for i in range(len(logger.losses)-len(last_losses), len(logger.losses))])
            plt.ylabel('Loss')
            plt.title("Loss for last 30 epochs")
            plt.legend()
            plt.savefig(f'{config["training"]["log_dir"]}/loss_recent.png')
            plt.close()
                        
            plt.plot(logger.train_accuracy, label='Train Accuracy')
            plt.plot(logger.test_accuracy, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Train and Test Accuracy')
            plt.legend()
            plt.savefig(f'{config["training"]["log_dir"]}/train_test_accuracy.png')
            plt.close()
        
    return model

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str, help='The config file for this experiment')
    
    args = parser.parse_args()
    model = train_model(args.config_file)
    
    torch.save(model.state_dict(), 'model_end.pt')
    