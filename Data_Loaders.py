import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('./saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        # print("Total:")
        # print(len(self.data))
        # print("Amount of no collisions:")
        # print(len(self.data[self.data[:,-1] == 0]))
        # print("Amount of collisions:")
        # print(len(self.data[self.data[:,-1] == 1]))
        self.data = np.unique(self.data, axis=0)
        # self.data = self.data[(self.data[:,0] == 150)&(self.data[:,1] != 150)&(self.data[:,2] != 150)&(self.data[:,3] != 150)&(self.data[:,4] != 150)&(self.data[:,-1] == 0)]
        # print("------------After clean up----------")
        # print("Total:")
        # print(len(self.data))
        # print("Amount of no collisions:")
        # print(len(self.data[self.data[:,-1] == 0]))
        # print("Amount of collisions:")
        # print(len(self.data[self.data[:,-1] == 1]))
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        # print(self.normalized_data[0].astype('float32'))
        # print(self.normalized_data[0,:-1].astype('float32'))
        # print(self.normalized_data[0,-1].astype('float32'))
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        x = self.normalized_data[idx,:-1].astype('float32')
        y = self.normalized_data[idx,-1].astype('float32')
        return {'input': x, 'label': y}
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        dataset_size = len(self.nav_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = data.sampler.SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=train_sampler)
        self.test_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=test_sampler)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
