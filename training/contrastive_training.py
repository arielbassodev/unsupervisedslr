import sys

from torch import device

import BackboneConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from lsfb_transfo.models import  encoder
from lightly.loss import NTXentLoss
import torch.optim as optim
from lsfb_transfo.training import Scheduler
from lightning.pytorch import LightningModule
import Lars

class ContrastiveTraining():

    def __init__(self,model,epochs,train_loader):
        self.model = model
        self.criterion = NTXentLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        self.epochs = epochs
        self.scheduler =  Scheduler.LinearSchedulerWithWarmup(self.optimizer)
        self.train_loader = train_loader
        
    @staticmethod    
    def plot_loss(epoch_losses):
         plt.figure(figsize=(8, 6))
         plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.title('Training Loss per Epoch')
         plt.grid(True)
         plt.legend()
         plt.savefig('Training_loss_simclr_others_essai_4.png')
         plt.show()
    
                
    def train(self):
        epoch_losses = []
        for epoch in range(self.epochs):
           a = self.optimizer.param_groups[0]['lr']
           self.optimizer.zero_grad()
           print("actual lr",a)
           running_loss = 0.0
           for id, feature in enumerate(tqdm(self.train_loader)):
              left_hand, right_hand, pose = feature[0].to('cuda'), feature[1].to('cuda'), feature[2].to('cuda')
              z1, z2 =  self.model.forward(left_hand, right_hand, pose)
              loss = self.criterion(z1, z2)
              loss.backward()
              self.optimizer.step()
              running_loss += loss.item()
              epoch_loss = running_loss / len(self.train_loader)
           epoch_losses.append(epoch_loss)
           #self.scheduler.step()
           print("la loss contrastive",epoch_loss)
        self.plot_loss(epoch_losses)
        return self.model.backbone
        