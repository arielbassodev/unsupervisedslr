from tqdm import tqdm
import torch.nn as nn
import torch.optim 


class classifier(nn.Module):
    def __init__(self, model, num_class):
        super().__init__()
        self.model = model
        self.num_class = num_class
        self.fc        = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class),
        )
    
    def forward(self, x):
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.eval()
        x = self.model(x)
        x = self.fc(x)
        return x
         
        
class fine_tuning_trainer():
    
    def __init__(self,train_loader, model, num_class, epochs):
        self.epochs       = epochs
        self.model        = model
        self.num_class    = num_class
        self.train_loader = train_loader
        self.criterion    = nn.CrossEntropyLoss()
        self.classifier   = classifier(self.model, self.num_class).to('cuda')
        self.optimizer    = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
    def train(self):
      for i in range(self.epochs):
        epoch_losses = []
        cumultaive_loss = 0
        accuracy = []
        for feature, target in tqdm(self.train_loader):
            feature = feature.to('cuda')
            target  = target.to('cuda')
            
            # Vérification des étiquettes
            print("Unique targets:", torch.unique(target))
            print("Max target:", torch.max(target))
            print("Min target:", torch.min(target))
            
            # Assurez-vous que les étiquettes sont valides
            invalid_targets = target[(target < 0) | (target >= self.num_class)]
            assert invalid_targets.numel() == 0, f"Invalid target values: {invalid_targets}"
            
            embeddings = self.classifier(feature)
            loss = self.criterion(embeddings, target)
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            cumultaive_loss += loss.item()
        epoch_acc = fine_tuning_trainer.validation(self.model, self.train_loader)
        accuracy.append(epoch_acc)
        epoch_losses.append(loss.item())
        
    def validation(model, test_loader):
        acc      = 0
        model   = model.eval()
        for feature, target in tqdm(test_loader):
            embeddings = model(feature)
            logits = nn.Softmax(dim=1)(embeddings)
            preds  = torch.argmax(logits, dim=1)
            acc += torch.sum(preds == target).item()
        return acc / len(test_loader)
@staticmethod
def number_target(dataloader):
    unique_targets = set()
    for feature, target in tqdm(dataloader):
        unique_targets.update(target.tolist())
    num_class = len(unique_targets)
    return num_class





                
                
                
                
        
    
    