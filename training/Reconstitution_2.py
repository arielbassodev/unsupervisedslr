import sys
from lsfb_transfo.models import *
from lsfb_transfo.loader import *
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torch
from pytorch_metric_learning.losses import SupConLoss

class Reconstitution_2(nn.Module):
    def __init__(self, backbone):
        super(Reconstitution_2, self).__init__()
        self.backbone = backbone
        self.mlp      = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward(self, x,y,z):
        x = self.backbone(x,y,z)
        x = self.mlp(x)
        return x

class ReconstitutionModel_2():

    def projector(self,x):
        x =x.to('cuda')
        mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        mlp = mlp.to('cuda')
        return mlp(x)


    def permute_frames(self,x):
        batch_size, seq_len, num_points, dim = x.shape
        permuted_indices = torch.stack([torch.randperm(seq_len) for _ in range(batch_size)])
        x_permuted = x.clone()
        for i in range(batch_size):
            x_permuted[i] = x[i, permuted_indices[i]]
        return x_permuted
    def __init__(self, model, epoch, loader):
        self.model = model
        self.epoch = epoch
        self.train_loader = loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)

    def permute_first_last_frames(self, x,y, z):
        batch_size, seq_len, num_points, dim = x.shape
        num_frames = seq_len // 3
        # Vérifier que num_frames est valide
        # Générer des indices de permutation pour les premières et dernières frames
        permuted_indices_first = torch.stack([torch.randperm(num_frames) for _ in range(batch_size)])
        permuted_indices_last = torch.stack([torch.randperm(num_frames) for _ in range(batch_size)])

        # Appliquer la permutation aux premières et dernières frames
        x_permuted = x.clone()  # Travailler sur une copie de x
        y_permuted = y.clone()
        z_permuted = z.clone()
        for i in range(batch_size):
            # Permuter les premières frames
           #x_permuted[i, :num_frames] = x[i, permuted_indices_first[i]]
           #y_permuted[i, :num_frames] = y[i, permuted_indices_first[i]]
           #z_permuted[i, :num_frames] = z[i, permuted_indices_first[i]]
            # Permuter les dernières frames
           x_permuted[i, -num_frames:] = x[i, seq_len - num_frames + permuted_indices_last[i]]
           y_permuted[i, -num_frames:] = y[i, seq_len - num_frames + permuted_indices_last[i]]
           z_permuted[i, -num_frames:] = z[i, seq_len - num_frames + permuted_indices_last[i]]

        return x_permuted.to(torch.float32), y_permuted.to(torch.float32), z_permuted.to(torch.float32)

    def generate_inputs(self, x, y, z):
        #x1 = x.clone()
        #indices = torch.randint(3, 26, (6,))
        #n1, n2, n3, n4, n5, n6 = indices.tolist()
        #x1[:, [n1, n2, n3,n4, n5,n6]] = x1[:, [n3,n6,n5,n2,n4,n1,n6]]
        #x1 = self.permute_first_last_frames(x)
        x1 = self.permute_first_last_frames(x, y, z)
        return x1


    def plot_loss(self,epoch_losses):
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
        for epoch in range(self.epoch):
            running_loss = 0.0
            for id, feature in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                left_hand, right_hand, pose = feature[0].to('cuda'), feature[1].to('cuda'), feature[2].to('cuda')
                #new_left_hand  = self.generate_inputs(left_hand).to(torch.float32)
                #new_right_hand = self.generate_inputs(right_hand).to(torch.float32)
                #new_pose       = self.generate_inputs(pose).to(torch.float32)
                new_left_hand, new_right_hand, new_pose = self.generate_inputs(left_hand, right_hand, pose)
                nl, nr, np = self.generate_inputs(left_hand, right_hand, pose)
                #nl = self.generate_inputs(left_hand).to(torch.float32)
                #nr = self.generate_inputs(right_hand).to(torch.float32)
                #np = self.generate_inputs(pose).to(torch.float32)
                aug_sign       = torch.cat((new_left_hand, new_right_hand, new_pose), dim=2)
                z1 = self.model(left_hand.to(torch.float32), right_hand.to(torch.float32),pose.to(torch.float32))
                zr = self.model(nl.to(torch.float32), nr.to(torch.float32), np.to(torch.float32))
                z2 = self.model(new_left_hand, new_right_hand,new_pose)
                #loss1 = self.criterion(z1, z2)
                loss2 = self.criterion(z2, zr)
                predictor = self.projector(z1)
                loss3 = self.criterion(predictor, z2.detach())
                loss =  loss2 + 0.0022*loss3
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_loss = running_loss / len(self.train_loader)
            epoch_losses.append(epoch_loss)
            # self.scheduler.step()
            print("la loss", epoch_loss)
        self.plot_loss(epoch_losses)
        return self.model.backbone



