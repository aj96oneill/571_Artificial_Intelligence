from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 50
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    losses = {}
    loss_function = nn.BCELoss()
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    # losses.append(min_loss)

    training_loss = {}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            output = model(sample['input'])
            label = sample['label'].float()
            loss = loss_function(output, label.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        loss = model.evaluate(model, data_loaders.train_loader, loss_function)
        if loss < min_loss:
            min_loss = loss
        training_loss[str(epoch_i)] = loss/no_epochs
        loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses[str(epoch_i)] = loss/no_epochs
            
    torch.save(model.state_dict(), "./saved/saved_model.pkl", _use_new_zipfile_serialization=False)

    plt.plot(list(training_loss.keys()),list(training_loss.values()), label="train")
    plt.plot(list(losses.keys()),list(losses.values()), label="test")
    plt.show()


if __name__ == '__main__':
    no_epochs = 70
    train_model(no_epochs)
