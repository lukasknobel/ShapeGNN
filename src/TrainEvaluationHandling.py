import statistics
import time

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from src.util import set_seed, accuracy


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomBatchSampler(Sampler):
    def __init__(self, indices, random_sampling, batch_size):
        self.indices = indices
        self.random_sampling = random_sampling
        self.batch_size = batch_size

    def __iter__(self):
        if self.random_sampling:
            used_idxs = torch.randperm(self.indices.shape[0])
        else:
            used_idxs = torch.arange(self.indices.shape[0])

        if used_idxs.shape[0] - self.batch_size < 1:
            # in case there is at most one batch
            yield self.indices[used_idxs]
        else:
            i = 0
            for i in range(0, used_idxs.shape[0] - self.batch_size, self.batch_size):
                yield self.indices[used_idxs[i:i + self.batch_size]]
            yield self.indices[used_idxs[i + self.batch_size:]]

    def __len__(self):
        return (self.indices.shape[0] + self.batch_size - 1) // self.batch_size


def train_model(model, optimizer, scheduler, batch_size, epochs, patience, trainval_dataset, train_idxs,
                val_idxs, checkpoint_name, summary_writer, device, graph_of_graphs=False, label_smoothing=0.0,
                clip_norm=0.0, return_last=False, num_workers=0, disable_tqdm=False):

    # load the datasets
    train_dataloader = DataLoader(trainval_dataset, sampler=CustomBatchSampler(indices=train_idxs, random_sampling=True,
        batch_size=batch_size), num_workers=num_workers, collate_fn=lambda x: x, batch_size=None, pin_memory=True)
    if val_idxs is None:
        val_dataloader = None
    else:
        val_dataloader = DataLoader(trainval_dataset,
                                    sampler=CustomBatchSampler(indices=val_idxs, random_sampling=False,
                                                               batch_size=batch_size),
                                    num_workers=num_workers, collate_fn=lambda x: x, batch_size=None, pin_memory=True)

    # loss module
    loss_module = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    train_losses = []
    train_accs_1 = []
    train_accs_5 = []
    val_losses = []
    val_accs_1 = []
    val_accs_5 = []
    best_model_epoch = None
    num_epoch_digits = len(str(epochs))

    start = time.time()

    epoch_times = []
    val_times = []

    for epoch in range(epochs):
        running_losses = AverageMeter('Loss')
        running_train_acc_1 = AverageMeter('Accuracy_1')
        running_train_acc_5 = AverageMeter('Accuracy_5')
        epoch_descr = f'Epoch {str(epoch + 1).zfill(num_epoch_digits)}/{epochs}'

        model.train()
        for batch_data in tqdm(train_dataloader, leave=False, desc=epoch_descr, disable=disable_tqdm):
            # training
            batch_data = batch_data.to(device, non_blocking=True)
            if graph_of_graphs:
                preds = model(x=batch_data.x, adj_t=batch_data.adj_t, pos=batch_data.pos, batch=batch_data.batch, 
                    sub_x=batch_data.sub_x, sub_adj_t=batch_data.sub_adj_t, sub_batch=batch_data.sub_batch,
                    edge_index=batch_data.edge_index, batch_lengths=batch_data.batch_lengths,
                    edge_batch=batch_data.edge_batch)
            else:
                preds = model(x=batch_data.x, adj_t=batch_data.adj_t, pos=batch_data.pos, batch=batch_data.batch,
                    edge_index=batch_data.edge_index, batch_lengths=batch_data.batch_lengths,
                    edge_batch=batch_data.edge_batch)
            loss = loss_module(preds, batch_data.y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_norm != 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            # training statistics
            running_losses.update(loss.item(), batch_size)
            accs = accuracy(preds, batch_data.y)
            running_train_acc_1.update(accs[0], batch_size)
            running_train_acc_5.update(accs[1], batch_size)

        train_losses.append(running_losses.avg)
        train_accs_1.append(running_train_acc_1.avg)
        train_accs_5.append(running_train_acc_5.avg)

        print(epoch_descr, end='\t')

        print(f'Train loss: {round(float(running_losses.avg), 3): <8}', end='')

        # train accuracy
        print(f'Train 1: {round(float(running_train_acc_1.avg), 3): <8}', end='')
        print(f'Train 5: {round(float(running_train_acc_5.avg), 3): <8}', end='')

        # validation
        if val_dataloader is not None:
            val_loss, val_accs, val_time = evaluate_model(model, val_dataloader, graph_of_graphs, label_smoothing, device)
            val_losses.append(val_loss)
            val_accs_1.append(val_accs[0])
            val_accs_5.append(val_accs[1])
            val_times.append(val_time)
            print(f'Val loss: {round(float(val_loss), 3): <8}', end='')
            print(f'Val 1: {round(float(val_accs[0]), 3): <8}', end='')
            print(f'Val 5: {round(float(val_accs[1]), 3): <8}', end='')

        epoch_times.append(time.time() - start)
        print(f'{round(epoch_times[-1], 1):<5}', end='')

        # add metrics to the summary writer
        if val_dataloader is None:
            summary_writer.add_scalars('Loss', {'train': running_losses.avg}, epoch)
            summary_writer.add_scalars('Accuracy', {'train': running_train_acc_1.avg}, epoch)
            summary_writer.add_scalars('Accuracy-5', {'train': running_train_acc_5.avg}, epoch)
        else:
            summary_writer.add_scalars('Loss', {'train': running_losses.avg, 'val': val_loss}, epoch)
            summary_writer.add_scalars('Accuracy', {'train': running_train_acc_1.avg, 'val': val_accs_1[-1]}, epoch)
            summary_writer.add_scalars('Accuracy-5', {'train': running_train_acc_5.avg, 'val': val_accs_5[-1]}, epoch)
        summary_writer.add_scalars('Time', {'train': epoch_times[-1]}, epoch)
        summary_writer.add_scalars('LR', {'train': optimizer.param_groups[0]['lr']}, epoch)

        # save (current) best model if a validation set is used, otherwise save the latest
        if val_dataloader is None:
            best_model_epoch = epoch
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoint_name)
        else:
            if len(val_accs_1) == 1 or val_accs_1[-1] > val_accs_1[best_model_epoch]:
                print('*', end='')
                best_model_epoch = epoch
                state_dict = model.state_dict()
                torch.save(state_dict, checkpoint_name)
            elif 0 < patience <= len(val_accs_1) and val_accs_1[best_model_epoch] > max(val_accs_1[-patience:]):
                print(f'Early stopping')
                break
        print()
        if scheduler is not None:
            scheduler.step(epoch)
        start = time.time()

    print(f'Median time per epoch: {statistics.median(epoch_times):.2f}s')
    if val_dataloader is not None:
        print(f'Median time per validation: {statistics.median(val_times):.2f}s')

    # save losses and accuracies
    if val_dataloader is None:
        results = np.vstack([np.array(train_losses), torch.stack(train_accs_1).detach().cpu().numpy(), torch.stack(train_accs_5).detach().cpu().numpy(), epoch_times])
    else:
        results = np.vstack([np.array(train_losses), torch.stack(train_accs_1).detach().cpu().numpy(), torch.stack(train_accs_5).detach().cpu().numpy(),
            torch.stack(val_accs_1).detach().cpu().numpy(), torch.stack(val_accs_5).detach().cpu().numpy(), epoch_times])
    np.savetxt(f'{checkpoint_name.split(".", 1)[0]}_results.csv', results, '%f')

    if return_last:
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_name)
    else:
        # load best model and return it.
        state_dict = torch.load(checkpoint_name)
        model.load_state_dict(state_dict)

    return model


def evaluate_model(model, data_loader, graph_of_graphs, label_smoothing, device):
    model.eval()

    # loss module
    loss_module = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    running_losses = AverageMeter('Loss')
    running_val_acc_1 = AverageMeter('Accuracy_1')
    running_val_acc_5 = AverageMeter('Accuracy_5')
    with torch.no_grad():
        start = time.time()
        for batch_data in data_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            if graph_of_graphs:
                preds = model(x=batch_data.x, adj_t=batch_data.adj_t, pos=batch_data.pos, batch=batch_data.batch, 
                    sub_x=batch_data.sub_x, sub_adj_t=batch_data.sub_adj_t, sub_batch=batch_data.sub_batch,
                    edge_index=batch_data.edge_index, batch_lengths=batch_data.batch_lengths,
                    edge_batch=batch_data.edge_batch)
            else:
                preds = model(x=batch_data.x, adj_t=batch_data.adj_t, pos=batch_data.pos, 
                    batch=batch_data.batch, edge_index=batch_data.edge_index, batch_lengths=batch_data.batch_lengths,
                    edge_batch=batch_data.edge_batch)

            loss = loss_module(preds, batch_data.y)

            # training statistics
            running_losses.update(loss.item(), preds.shape[0])

            accs = accuracy(preds, batch_data.y)
            running_val_acc_1.update(accs[0], preds.shape[0])
            running_val_acc_5.update(accs[1], preds.shape[0])
        eval_time = time.time() - start
    return running_losses.avg, (running_val_acc_1.avg, running_val_acc_5.avg), eval_time


def test_model(model, batch_size, test_dataset, graph_of_graphs, label_smoothing, device, seed, num_workers=2):
    set_seed(seed)

    test_dataloader = DataLoader(test_dataset,
                                 sampler=CustomBatchSampler(indices=torch.arange(len(test_dataset)),
                                                            random_sampling=False, batch_size=batch_size),
                                 num_workers=num_workers, collate_fn=lambda x: x, batch_size=None, pin_memory=True)

    loss, accs, test_time = evaluate_model(model, test_dataloader, graph_of_graphs, label_smoothing, device)
    test_results = {
        'loss': loss,
        'accuracy-1': float(accs[0].detach().cpu().numpy()),
        'accuracy-5': float(accs[1].detach().cpu().numpy()),
        'time': test_time
    }
    return test_results