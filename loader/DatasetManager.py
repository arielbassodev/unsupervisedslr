from torch.utils.data import random_split
import lsfb_transfo.loader.load_data as load_data


class DatasetManager():
     def __init__(self, dataset):
         self.dataset = dataset
         
     def split_dataset(trainset, testset):
          dataset_size = len(trainset)
          unsupervised_size = int(2*dataset_size / 3)
          supervised_size = dataset_size - unsupervised_size
          unsupervised_dataset, supervised_dataset = random_split(trainset, [unsupervised_size, supervised_size])
          unsupervised_dataset = load_data.CustomDataset.build_dataset(unsupervised_dataset)
          supervised_dataset = load_data.CustomDataset.build_dataset(supervised_dataset)
          testset = load_data.CustomDataset.build_dataset(testset)
          return unsupervised_dataset, supervised_dataset, testset

