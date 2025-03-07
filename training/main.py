import os
import sys
import torch
import torch.nn as nn
from click.core import F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
from tqdm import tqdm
from pytorch_metric_learning import losses
import contrastive_training
import torch.optim as optim
import BackboneConfig
import fine_tuning_trainer
import Scheduler
from lightly.loss import NTXentLoss
import matplotlib.pyplot as plt
import Reconstitution_2
import reconstitution
from lsfb_transfo.loader import load_data
from lsfb_transfo.models import simclr, encoder
#from lsfb_transfo.training.Reconstitution_2 import Reconstitution_2
from lsfb_transfo.training.reconstitution import Reconstitution

trainset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/LSFB_Dataset/lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "train",
    n_labels= 500,
    sequence_max_length=30,
    show_progress=True,
))

testset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/LSFB_Dataset/lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "test",
    n_labels=500,
    sequence_max_length=30,
    show_progress=True,
))


train_loader =  load_data.CustomDataset.build_dataset(trainset)
unsup_loader =  load_data.CustomDataset.proprocess_data(trainset)
test_loader = load_data.CustomDataset.build_dataset(testset)
config =  BackboneConfig.BackboneConfig()
backbone = encoder.ViTModel(**vars(config)).to('cuda')
#model  = simclr.SimCLR(backbone).to('cuda')
#bk = contrastive_training.ContrastiveTraining(model,50,train_loader).train()
bk_1 = Reconstitution(backbone).to('cuda')
bk = reconstitution.ReconstitutionModel(bk_1,50,train_loader).train()

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        for params in self.backbone.parameters():
           params.requires_grad = False
        self.fc = nn.Linear(500, 500)
        self.mlp = nn.Sequential(
            #nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 500),
        )
   def forward(self, x1,y,z1):
      x = self.backbone(x1,y,z1)
      x = self.mlp(x)
      x = self.fc(x)
      return x
cl = classifier(bk, 500).to('cuda')
import pytorch_lightning as L
import torchmetrics as TM
import torch
from torch import nn, optim
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = cl
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.train_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.train_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.train_top5_acc(logits, targets)
        self.train_top3_acc(logits, targets)
        self.train_top10_acc(logits, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_top10_acc', self.train_top5_acc, on_step=True, on_epoch=True)
        self.log('train_top3_acc', self.train_top3_acc, on_step=True, on_epoch=True)
        self.log('train_top5_acc', self.train_top10_acc, on_step=True, on_epoch=True)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_top3_acc(logits, targets)
        self.val_top10_acc(logits, targets)
        self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_top5_acc', self.val_top5_acc, on_step=True, on_epoch=True)
        self.log('val_top3_acc', self.val_top3_acc, on_step=True, on_epoch=True)
        self.log('val_top10_acc', self.val_top10_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
my_module = Module()
trainer = L.Trainer(max_epochs=1000)
trainer.fit(
   my_module,
   train_loader,
   test_loader
)

backbone2 = encoder.ViTModel(**vars(config)).to('cuda')
#model  = simclr.SimCLR(backbone).to('cuda')
#bk = contrastive_training.ContrastiveTraining(model,30,train_loader).train()
bk_12 = Reconstitution(backbone2).to('cuda')
bk2 = reconstitution.ReconstitutionModel(bk_12,50,train_loader).train()

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        for params in self.backbone.parameters():
           params.requires_grad = False
        self.fc = nn.Linear(500, 500)
        self.mlp = nn.Sequential(
            #nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 500),
        )
   def forward(self, x1,y,z1):
      x = self.backbone(x1,y,z1)
      x = self.mlp(x)
      x = self.fc(x)
      return x
cl2 = classifier(bk2, 500).to('cuda')
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = cl2
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.train_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.train_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.train_top5_acc(logits, targets)
        self.train_top3_acc(logits, targets)
        self.train_top10_acc(logits, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_top10_acc', self.train_top5_acc, on_step=True, on_epoch=True)
        self.log('train_top3_acc', self.train_top3_acc, on_step=True, on_epoch=True)
        self.log('train_top5_acc', self.train_top10_acc, on_step=True, on_epoch=True)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_top3_acc(logits, targets)
        self.val_top10_acc(logits, targets)
        self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_top5_acc', self.val_top5_acc, on_step=True, on_epoch=True)
        self.log('val_top3_acc', self.val_top3_acc, on_step=True, on_epoch=True)
        self.log('val_top10_acc', self.val_top10_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
my_module = Module()
trainer = L.Trainer(max_epochs=1000)
trainer.fit(
   my_module,
   train_loader,
   test_loader
)


backbone3 = encoder.ViTModel(**vars(config)).to('cuda')
#model  = simclr.SimCLR(backbone).to('cuda')
#bk = contrastive_training.ContrastiveTraining(model,30,train_loader).train()
bk_3 = Reconstitution_2.Reconstitution_2(backbone3).to('cuda')
bk3 = Reconstitution_2.ReconstitutionModel_2(bk_3,50,train_loader).train()

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        for params in self.backbone.parameters():
           params.requires_grad = False
        self.fc = nn.Linear(500, 500)
        self.mlp = nn.Sequential(
            #nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 500),
        )
   def forward(self, x1,y,z1):
      x = self.backbone(x1,y,z1)
      x = self.mlp(x)
      x = self.fc(x)
      return x
cl3 = classifier(bk3, 500).to('cuda')
import pytorch_lightning as L
import torchmetrics as TM
import torch
from torch import nn, optim
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = cl3
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.train_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.train_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.train_top5_acc(logits, targets)
        self.train_top3_acc(logits, targets)
        self.train_top10_acc(logits, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_top10_acc', self.train_top5_acc, on_step=True, on_epoch=True)
        self.log('train_top3_acc', self.train_top3_acc, on_step=True, on_epoch=True)
        self.log('train_top5_acc', self.train_top10_acc, on_step=True, on_epoch=True)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_top3_acc(logits, targets)
        self.val_top10_acc(logits, targets)
        self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_top5_acc', self.val_top5_acc, on_step=True, on_epoch=True)
        self.log('val_top3_acc', self.val_top3_acc, on_step=True, on_epoch=True)
        self.log('val_top10_acc', self.val_top10_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
my_module = Module()
trainer = L.Trainer(max_epochs=1000)
trainer.fit(
   my_module,
   train_loader,
   test_loader
)


backbone4 = encoder.ViTModel(**vars(config)).to('cuda')
#model  = simclr.SimCLR(backbone).to('cuda')
#bk = contrastive_training.ContrastiveTraining(model,30,train_loader).train()
bk_4 = Reconstitution_2.Reconstitution_2(backbone4).to('cuda')
bk4 = Reconstitution_2.ReconstitutionModel_2(bk_4,50,train_loader).train()

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        for params in self.backbone.parameters():
           params.requires_grad = False
        self.fc = nn.Linear(500, 500)
        self.mlp = nn.Sequential(
            #nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 500),
        )
   def forward(self, x1,y,z1):
      x = self.backbone(x1,y,z1)
      x = self.mlp(x)
      x = self.fc(x)
      return x
cl4 = classifier(bk4, 500).to('cuda')
import pytorch_lightning as L
import torchmetrics as TM
import torch
from torch import nn, optim
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = cl4
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.train_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.train_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.train_top5_acc(logits, targets)
        self.train_top3_acc(logits, targets)
        self.train_top10_acc(logits, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_top10_acc', self.train_top5_acc, on_step=True, on_epoch=True)
        self.log('train_top3_acc', self.train_top3_acc, on_step=True, on_epoch=True)
        self.log('train_top5_acc', self.train_top10_acc, on_step=True, on_epoch=True)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_top3_acc(logits, targets)
        self.val_top10_acc(logits, targets)
        self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_top5_acc', self.val_top5_acc, on_step=True, on_epoch=True)
        self.log('val_top3_acc', self.val_top3_acc, on_step=True, on_epoch=True)
        self.log('val_top10_acc', self.val_top10_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
my_module = Module()
trainer = L.Trainer(max_epochs=1000)
trainer.fit(
   my_module,
   train_loader,
   test_loader
)