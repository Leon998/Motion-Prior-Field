import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset_config import *
from utils.object_config import objects

object_cls = objects['mug']
output_dim = object_cls.g_clusters * len(object_cls.grasp_types)
position_active = False
if position_active:
    input_dim = 3
    writer = SummaryWriter("classify/tensorbd/noisy_" + object_cls.name + "/position_" + str(object_cls.g_clusters) + "_manifolds/")
else:
    input_dim = 7
    writer = SummaryWriter("classify/tensorbd/noisy_" + object_cls.name + "/pose_" + str(object_cls.g_clusters) + "_manifolds/")

MLP = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, output_dim)
)


def train(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        writer.add_scalar('Loss/train', loss, t)
        writer.add_scalar('Learning rate/train', optimizer.state_dict()['param_groups'][0]['lr'], t)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss = loss
    return train_loss

def t_test(dataloader, model, loss_fn, t):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    writer.add_scalar('Accuracy/test', correct, t)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    torch.manual_seed(1)  # reproducible

    path = 'classify/training_data/' + object_cls.name + '_field.txt'
    batch_size = 256
    epochs = 200
    print('Current clusters:', object_cls.g_clusters)
    # Get cpu or gpu device for training.
    device = "cuda"
    print(f"Using {device} device")

    _, _, train_dataloader, test_dataloader = data_loading(path, batch_size, position_active)

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = MLP.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=15,
                                                           verbose=True, eps=0.0005)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, t)
        t_test(test_dataloader, model, loss_fn, t)
        scheduler.step(train_loss)
    print("Done!")
    writer.close()
    # Save model
    if position_active:
        torch.save(model, 'classify/trained_models/' + object_cls.name + '/noisy_position_' + str(object_cls.g_clusters) +'.pkl')
    else:
        torch.save(model, 'classify/trained_models/' + object_cls.name + '/noisy_pose_' + str(object_cls.g_clusters) +'.pkl')
