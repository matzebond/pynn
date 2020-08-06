#!/usr/bin/env python3

import sys
[sys.path.append(i) for i in ['.', '..', '../..']]
import argparse
import numpy as np
import torch

from pynn.io.kaldi_seq import ScpStreamReader, ScpBatchReader
from pynn.net.lid_markus import LidMarkus, LidMarkusLinear
from pynn.trainer.markus_trainer import train_model, test_model

parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--train-scp', help='path to train scp', required=True)
parser.add_argument('--train-target', help='path to train target', required=True)
parser.add_argument('--valid-scp', help='path to validation scp', required=True)
parser.add_argument('--valid-target', help='path to validation target', required=True)
parser.add_argument('--test-scp', help='path to test scp', required=True)
parser.add_argument('--test-target', help='path to test target', required=True)

parser.add_argument('--n-classes', type=int, required=True)
parser.add_argument('--input-dim', type=int, help='input dimensions', required=True)
parser.add_argument('--model-path', help='model saving path', default='model')

parser.add_argument('--n-epoch', type=int, default=50)
# parser.add_argument('--batch', help='batch mode', action='store_true')
parser.add_argument('--batch-size', help='number of inputs per batch', type=int, default=1024)
parser.add_argument('--shuffle', help='shuffle samples every epoch', action='store_true')
parser.add_argument('--optimizer-name', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop'])
parser.add_argument('--optimizer-momentum', type=float, default=0.9)
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--use-scheduler', action='store_true')
parser.add_argument('--scheduler', type=str, default='expLR', choices=['redLROnPlateau', 'expLR'])
parser.add_argument('--expLR-decay-rate', type=float, default=.98)
parser.add_argument('--weight-decay', type=float, default=0.0)

parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--max-len', help='max sequence length', type=int, default=5000)
parser.add_argument('--max-utt', help='max utt per partition', type=int, default=-1)
parser.add_argument('--mean-sub', help='mean subtraction', default=False)
parser.add_argument('--zero-pad', help='padding zeros to sequence end', type=int, default=0)
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument('--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument('--time-stretch', help='argument inputs', action='store_true')
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)

parser.add_argument('--utt-frame-length', type=int, default=15)
parser.add_argument('--utt-stride', type=int, default=10)
parser.add_argument('--utt-spread', type=int, default=3)

# parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
# parser.add_argument('--n-const', help='constant steps', type=int, default=0)
# parser.add_argument('--n-print', help='inputs per update', type=int, default=5000)
parser.add_argument('--fp16', help='fp16 or not', default=False)




if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # model = LidMarkus(n_classes=args.n_classes,
    #                   input_size=args.input_dim,
    #                   utt_frame_length=args.utt_frame_length).to(device)

    model = LidMarkusLinear(n_classes=args.n_classes,
                            input_size=args.input_dim,
                            utt_frame_length=args.utt_frame_length).to(device)

    # ScpReader = ScpBatchReader if args.batch else ScpStreamReader
    tr_reader = ScpStreamReader(args.train_scp, args.train_target, sek=False,
                                sort_src=True, max_len=args.max_len, max_utt=args.max_utt,
                                mean_sub=args.mean_sub, zero_pad=args.zero_pad,
                                fp16=args.fp16, shuffle=args.shuffle,
                                spec_drop=args.spec_drop, spec_bar=args.spec_bar,
                                time_stretch=args.time_stretch, time_win=args.time_win)
    cv_reader = ScpStreamReader(args.valid_scp, args.valid_target, sek=False,
                                sort_src=True, max_len=args.max_len, max_utt=args.max_utt,
                                mean_sub=args.mean_sub, zero_pad=args.zero_pad, fp16=args.fp16)
    test_reader = ScpStreamReader(args.test_scp, args.test_target, sek=False,
                                  sort_src=True, max_len=args.max_len, max_utt=args.max_utt,
                                  mean_sub=args.mean_sub, zero_pad=args.zero_pad, fp16=args.fp16)

    cfg = {'model_path': args.model_path, 'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
           'optimizer_name': args.optimizer_name, 'optimizer_momentum': args.optimizer_momentum, 'input_dim': args.input_dim,
           'use_scheduler': args.use_scheduler, 'scheduler': args.scheduler, 'expLR_decay_rate': args.expLR_decay_rate,
           'utt_stride': args.utt_stride, 'utt_frame_length': args.utt_frame_length, 'utt_spread': args.utt_spread}

    datasets = (tr_reader, cv_reader)
    training_loss, training_accuracy, val_loss, val_accuracy = train_model(model, datasets, args.n_epoch, device, cfg)
    test_loss, test_acc = test_model(model, test_reader, device, cfg)

    training_loss = np.array(training_loss).reshape(1, -1)
    training_accuracy = np.array(training_accuracy).reshape(1, -1)
    val_loss = np.array(val_loss).reshape(1, -1)
    val_accuracy = np.array(val_accuracy).reshape(1, -1)

    np.savetxt('{}/{}'.format(args.model_path, 'training_loss.csv'), training_loss, delimiter=',', fmt='%.9e')
    np.savetxt('{}/{}'.format(args.model_path, 'training_accuracy.csv'), training_accuracy, delimiter=',', fmt='%.9e')
    np.savetxt('{}/{}'.format(args.model_path, 'val_loss.csv'), val_loss, delimiter=',', fmt='%.9e')
    np.savetxt('{}/{}'.format(args.model_path, 'val_accuracy.csv'), val_accuracy, delimiter=',', fmt='%.9e')
    np.savetxt('{}/{}'.format(args.model_path, 'test_loss_acc.csv'), np.array([test_loss, test_acc]).reshape(1, -1),
               delimiter=',', fmt='%.9e', header='test_loss,test_acc')
