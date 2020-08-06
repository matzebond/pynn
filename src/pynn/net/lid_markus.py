# extracted feature frames are 32ms window with a 10ms frame-shift
# use a context of +/- 7 frames with a spread of 3

# ML-BNF network features 7 layers with 1,600
# DNN for the acoustic model featured 6 layers with 1,000 neurons each.
# second to last layer was the bottleneck layer with only 42 neurons.

# first greedy layer-wise pre-training to initialize the weights.

# then fine-tune the  network

# Newbob learning rate scheduling with an initial learning rate of 1.0
# was applied, starting the exponential learning rate decay if the frame error rate of
# the validation set decreased by less than 0.005 between two epochs.
# stopped when the FER decreased by less than 0.0001 between two epochs in the exponential phase.


# OLD
# pre-trained the network in the notion of a de-noising auto-encoder using Gaussian noise and a corruption rate of 0.2

import torch
import torch.nn as nn

class BottleneckFeature(nn.Module):
    def __init__(self, input_size, utt_frame_length, bottleneck_size):
        super(BottleneckFeature, self).__init__()

        # Markus: context with 7 -> kernel size 15
        # Draper: context of 11 -> kernel size 23

	# [b,c_in,l] -> [b,c_out,l']
  	# [b,42,~1200] -> [b, 1535, l']    kernel 15 * 40
	# kernel := in_channels * kernel_size
        self.conv_stack = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=1535, kernel_size=utt_frame_length, stride=1, padding=0, dilation=3),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=1535, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=1535, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=1535, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=1535, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=1535, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Tanh(),
                                        nn.Conv1d(in_channels=1535,       out_channels=bottleneck_size, kernel_size=1 , stride=1, padding=0, dilation=1),
                                        nn.Sigmoid()
        )

        # self.conv_post = nn.Sequential(nn.Conv1d(in_channels=bottleneck_size,   out_channels=1535, kernel_size=1 , stride=1, dilation=1),
        #                                nn.Tanh(),
        #                                nn.Conv1d(in_channels=1535, out_channels=input_size, kernel_size=1 , stride=1, dilation=1),
        #                                nn.Softmax(dim=-1))
        
    def forward(self, x, mask=None):
        x = x.transpose(1,2)
        x = self.conv_stack(x)
        return x

class AcousticModel(nn.Module):
    def __init__(self, n_classes, utt_frame_length, bottleneck_size):
        super(AcousticModel, self).__init__()
        # Draper: context of 11 -> kernel size 23
        self.lid_pre = nn.Sequential(nn.Conv1d(in_channels=bottleneck_size,   out_channels=1024, kernel_size=23, stride=1, dilation=3),
                                      nn.Tanh(),
                                      nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1 , stride=1, dilation=1),
                                      nn.Tanh(),
                                      nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1 , stride=1, dilation=1),
                                      nn.Tanh(),
                                      nn.Conv1d(in_channels=1024, out_channels=bottleneck_size, kernel_size=1 , stride=1, dilation=1),
                                      nn.Sigmoid()
        )
        self.lid_post = nn.Sequential(nn.Conv1d(in_channels=bottleneck_size,   out_channels=1024, kernel_size=1 , stride=1, dilation=1),
                                       nn.Tanh(),
                                       nn.Conv1d(in_channels=1024, out_channels=n_classes, kernel_size=1 , stride=1, dilation=1),
                                       nn.Softmax(dim=-1))
        
    def forward(self, x, mask=None):
        x = self.lid_pre(x)
        x = self.lid_post(x)
        # x = x.transpose(1,2)
        return x

class LidMarkus(nn.Module):
    def __init__(self, input_size, n_classes, utt_frame_length=15, bottleneck_size = 42):
        super(LidMarkus, self).__init__()

        self.featEx = BottleneckFeature(input_size, utt_frame_length, bottleneck_size)
        self.acousticModel = AcousticModel(n_classes, utt_frame_length, bottleneck_size)

    def forward(self, x, mask=None):
        x = self.featEx(x, mask=mask);
        x = self.acousticModel(x, mask=mask)
        return x






class LidMarkusLinear(nn.Module):
    def __init__(self, input_size, n_classes, utt_frame_length=15, bottleneck_size = 42):
        super(LidMarkusLinear, self).__init__()

        self.input_size = input_size
        self.utt_frame_length = utt_frame_length

        # go from size 1600  to 1535 = 1024 + 512 = 3 * 2^9
        self.bnf = nn.Sequential(nn.Linear(utt_frame_length * input_size, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 1535),
                                 nn.Tanh(),
                                 nn.Linear(1535, 42),
                                 nn.Sigmoid())

        # self.bnf_post = nn.Sequential(nn.Linear(42, 1535),
        #                               nn.Tanh(),
        #                               nn.Linear(1535, utt_frame_length * input_size))

        # go from size 1000  to 1024
        self.lid_pre = nn.Sequential(nn.Linear(42, 1024),
                                         nn.Tanh(),
                                         nn.Linear(1024, 1024),
                                         nn.Tanh(),
                                         nn.Linear(1024, 1024),
                                         nn.Tanh(),
                                         nn.Linear(1024, bottleneck_size),
                                         nn.Sigmoid())

        self.lid_post = nn.Sequential(nn.Linear(bottleneck_size, 1024),
                                          nn.Tanh(),
                                          nn.Linear(1024, n_classes),
                                          nn.Softmax(dim=-1)) # Markus uses Sigmoid


        # bnf: 14 * 40 * 1535 + 5 * 1535^2 + 1534 * 42 ~= 900k
        # lid: 42 * 1024 + 2 * 1024^2 + 2 * 1042 * 42 + 1024 * 10 ~= 118k
        # overall: ~ 1mio parameters

    def forward(self, x):
        x = x.view(-1, self.bnf[0].in_features)
        x = self.bnf(x)
        x = self.lid_pre(x)
        x = self.lid_post(x)
        return x

    def bnf_autoencoder(self, x):
        x = x.view(-1, self.bnf[0].in_features)
        x = self.bnf(x)
        x = self.bnf_post(x)
        x = v.view(-1, self.utt_frame_length)
        return x
