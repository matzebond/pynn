import time
import os
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from . import EpochPool, load_last_chkpt, save_last_chkpt

def bnf_epoch(model, data, device, batch_size, utt_frame_length, utt_stride, model_path, optimizer=None):
    data.initialize()
    n_utterances = len(data.label_dic)
    n_batches = math.ceil(n_utterances/batch_size)
    total_loss = 0.0
    total_n_correct_pred = 0
    total_n_labels = 0
    n_batch = 1
    while data.available():
        if optimizer: optimizer.zero_grad()

        src_seq, _ = data.next_batch2(utt_frame_length, utt_stride, batch_size, model_path)
        src_seq = src_seq.to(device)

        try:
            pred = model.bnf_autoencoder(src_seq)
            loss = torch.sqrt(nn.MSELoss()(pred, src_seq) + 1e-10) # why sqrt & eps

            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            print("==> (BNF {}) [batch {}/{}] [loss {}] [accuracy {:.2f}%]".format("Training" if optimizer else "Validation",
                                                                               n_batch, n_batches, loss.item(),
                                                                               100*n_correct_pred/n_labels))
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print("==> WARNING: ran out of memory on GPU at batch {}/{}".format(n_batch, n_batches))
                torch.cuda.empty_cache()
            raise err

        n_batch += 1

    return total_loss


def epoch(model, data, device, batch_size, utt_frame_length, utt_stride, model_path, optimizer=None):
    data.initialize()
    n_utterances = len(data.label_dic)
    n_batches = math.ceil(n_utterances/batch_size)
    total_loss = 0.0
    total_n_correct_pred = 0
    total_n_labels = 0
    n_batch = 1
    while data.available():
        if optimizer: optimizer.zero_grad()

        batch = data.next_batch(batch_size)
        src_seq, src_mask, target_seq = map(lambda x: x.to(device), batch)

        try:
            pred = model(src_seq, src_mask)
            label = torch.argmax(target_seq, dim=-1)
            label = torch.unsqueeze(label, 1).expand(batch_size, pred.shape[-1])
            loss = nn.CrossEntropyLoss()(pred, label)

            if optimizer:
                loss.backward()
                optimizer.step()

            pred = torch.argmax(pred, dim=1).detach()
            total_loss += loss.item()
            n_labels = pred.numel()
            n_correct_pred = pred.eq(label).sum().item()
            total_n_labels += n_labels
            total_n_correct_pred += n_correct_pred
            print("==> ({}) [batch {}/{}] [loss {}] [accuracy {:.2f}%]".format("Training" if optimizer else "Validation",
                                                                               n_batch, n_batches, loss.item(),
                                                                               100*n_correct_pred/n_labels))
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print("==> WARNING: ran out of memory on GPU at batch {}/{}".format(n_batch, n_batches))
                torch.cuda.empty_cache()
            raise err

        n_batch += 1

    return total_loss, 100 * total_n_correct_pred/total_n_labels


def epoch_packed(model, data, device, batch_size, utt_frame_length, utt_stride, utt_spread, model_path=None, optimizer=None):
    data.initialize()
    n_utterances = len(data.label_dic)
    data.available()
    # n_batches = data.num_packed_spreaded_batches(utt_frame_length, utt_stride, utt_spread, batch_size)
    # print("~ {} batches".format(n_batches))
    total_loss = 0.0
    total_n_correct_pred = 0
    total_n_labels = 0
    n_batch = 1
    while data.available():
        if optimizer: optimizer.zero_grad()

        batch = data.next_batch2(utt_frame_length, utt_stride, batch_size, model_path)
        # batch = data.next_batch_packed_spreaded(utt_frame_length, utt_stride, utt_spread, batch_size)
        src_seq, target_seq = map(lambda x: x.to(device), batch)

        try:
            pred = model(src_seq)
            label = torch.argmax(target_seq, dim=-1)
            loss = nn.CrossEntropyLoss()(pred, label)

            if optimizer:
                loss.backward()
                optimizer.step()

            pred = torch.argmax(pred, dim=1).detach()
            total_loss += loss.item()
            n_labels = pred.numel()
            n_correct_pred = pred.eq(label).sum().item()
            total_n_labels += n_labels
            total_n_correct_pred += n_correct_pred
            print("==> ({}) [batch {}/{}] [size {}] [loss {}] [accuracy {:.2f}%]".format("Training" if optimizer else "Validation",
                                                                                         n_batch, n_batches, n_labels,
                                                                                         loss.item(), 100*n_correct_pred/n_labels))
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print("==> WARNING: ran out of memory on GPU at batch {}/{}".format(n_batch, n_batches))
                torch.cuda.empty_cache()
            raise err

        n_batch += 1
    return total_loss, 100 * total_n_correct_pred/total_n_labels

def pretrain_model(model, datasets, epochs, device, cfg):
    pass

def train_model(model, datasets, epochs, device, cfg):
    model_path = cfg['model_path']
    utt_frame_length = cfg['utt_frame_length']
    utt_stride = cfg['utt_stride']
    utt_spread = cfg['utt_spread']


    optimizer = None
    if cfg['optimizer_name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], eps=1e-4)
    elif cfg['optimizer_name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['optimizer_momentum'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_name'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['lr'], momentum=cfg['optimizer_momentum'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError('Unkown optimizer name: {}'.format(cfg['optimizer_name']))

    tr_data, cv_data = datasets
    tr_data = cv_data #TODO only for debugging
    print("\n\nusing cv data as traning data\n----ONLY FOR DEBUGGING PURPOSES------\n\n")
    
    pool = EpochPool(5)
    epoch_i, _ = load_last_chkpt(model_path, model, optimizer)
    # for p in optimizer.param_groups:
    #     p['lr'] = lr

    if cfg['use_scheduler']:
        if cfg['scheduler'] == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg['expLR_decay_rate'])
        elif cfg['scheduler'] == 'redLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=.1,
                                                                   min_lr=.0001, verbose=True)

    # if cfg['pretraining'] and epoch_i == 0:
    #     pretrain_model(model,dataselts)

    training_loss, training_accuracy = [], []
    val_loss, val_accuracy = [], []
    learing_rates = []

    while epoch_i < epochs:
        epoch_i += 1
        print('[Epoch', epoch_i, ']')

        start = time.time()

        model.train()
        tr_loss, tr_accuracy = epoch_packed(model, tr_data, device, cfg['batch_size'], utt_frame_length, utt_stride, utt_spread, model_path, optimizer=optimizer)
        print('(Training) [elapse: {} min] [training loss {}] [training accuracy {:.2f}%] [lr {}]'.format((time.time() - start) / 60,
                                                                                                          tr_loss, tr_accuracy, learing_rates[-1]))
        training_loss.append(tr_loss)
        training_accuracy.append(tr_accuracy)

        start = time.time()

        print("==> (Validation)")
        model.eval()
        with torch.no_grad():
            cv_loss, cv_accu = epoch_packed(model, cv_data, device, cfg['batch_size'], utt_frame_length, utt_stride, utt_spread, model_path)
        print('(Validation) [elapse: {} min] [val loss {}] [val accuracy {:.2f}%]'.format((time.time() - start) / 60, cv_loss, cv_accu))

        val_loss.append(cv_loss)
        val_accuracy.append(cv_accu)

        for param_groups in optimizer.param_groups:
            learing_rates.append(param_groups['lr'])

        log_loss_accu(training_loss, training_accuracy, val_loss, val_accuracy, learing_rates, model_path)

        if math.isnan(cv_loss): break

        model_file = os.path.join(model_path, 'epoch-{}.pt'.format(epoch_i))
        pool.save(cv_loss, model_file, model)
        save_last_chkpt(model_path, epoch_i, model, optimizer)

        if cfg['use_scheduler']:
            scheduler.step()


    return training_loss, training_accuracy, val_loss, val_accuracy

def log_epoch(training_loss, training_accuracy, val_loss, val_accuracy, learing_rates, model_path):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(training_loss)+1), training_loss, label='training loss', c='tab:blue')
    plt.plot(range(1, len(val_loss)+1), val_loss, label='validation loss', c='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(training_accuracy)+1), training_accuracy, label='training acc', c='k')
    plt.plot(range(1, len(val_accuracy)+1), val_accuracy, label='val acc', c='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'training_stats.png'))

    np.savetxt(os.path.join(model_path, 'training_loss.csv'), np.array(training_loss).reshape(1, -1), delimiter=',', fmt='%.9e')
    np.savetxt(os.path.join(model_path, 'training_accuracy.csv'), np.array(training_accuracy).reshape(1, -1), delimiter=',', fmt='%.9e')
    np.savetxt(os.path.join(model_path, 'val_loss.csv'), np.array(val_loss).reshape(1, -1), delimiter=',', fmt='%.9e')
    np.savetxt(os.path.join(model_path, 'val_accuracy.csv'), np.array(val_accuracy).reshape(1, -1), delimiter=',', fmt='%.9e')
    np.savetxt('{}/{}'.format(model_path, 'learing_rates.csv'), np.array(learing_rates).reshape(1, -1), delimiter=',', fmt='%.9e')


    
def test_model(model, data, device, cfg):
    ''' Epoch operation in evaluation phase '''

    start = time.time()

    print("==> (Testing)")
    model.eval()
    with torch.no_grad():
       test_loss, test_accu = epoch(model, data, device, cfg['batch_size'], utt_frame_length, utt_stride, model_path)

    print('(Test) [elapse: {} min] [test loss {}] [test accuracy: {:.2f}%]'.format((time.time() - start)/60,
                                                                                   test_loss, test_accu))

    return test_loss, test_accu
