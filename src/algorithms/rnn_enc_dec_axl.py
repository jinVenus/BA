import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

from .algorithm_utils import Algorithm, PyTorchUtils


class RNNED(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'RNN-ED', num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3,
                 hidden_size: int = 100, sequence_length: int = 20, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.rnned = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame, seq):
        # X.interpolate(inplace=True)
        # X.bfill(inplace=True)
        # data = X.values
        # index = np.arange(0, data.shape[0] - 49, 25, int)
        # sequences = [data[i:i + self.sequence_length] for i in index]
        #sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        sequences = seq
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.rnned = RNNEDModule(X.shape[1], self.hidden_size,
                                 self.n_layers, self.use_bias, self.dropout,
                                 seed=self.seed, gpu=self.gpu)
        self.to_device(self.rnned)
        optimizer = torch.optim.Adam(self.rnned.parameters(), lr=self.lr)

        self.rnned.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.rnned(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.rnned.zero_grad()
                loss.backward()
                optimizer.step()

        self.rnned.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.rnned(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)


    def predict(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        # index = np.arange(0, data.shape[0] - 49, 25, int)
        # sequences = [data[i:i + self.sequence_length] for i in index]
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.rnned.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = self.rnned(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = outputs[:-1]
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})
            outputs = np.nanmean(lattice, axis=0)

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})
            errors = np.nanmean(lattice, axis=0)

        return scores, errors, outputs


class RNNEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.RNN(self.n_features, self.hidden_size, batch_first=True,
                              num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.RNN(self.n_features, self.hidden_size, batch_first=True,
                              num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_())

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size) # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        outputs = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        inputs = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        out, dec_hidden = self.decoder(inputs, dec_hidden)

        output = self.hidden2output(out.squeeze(0))
        # for i in reversed(range(ts_batch.shape[1])):
        #     outputs[:, i, :] = self.hidden2output(out.squeeze(0))
        #     if self.training:
        #         out, dec_hidden = self.decoder(inputs[:, i].unsqueeze(1).float(), dec_hidden)
        #         out = out.reshape(out.shape[0], -1)
        #     else:
        #         out, dec_hidden = self.decoder(inputs[:, i].unsqueeze(1).float(), dec_hidden)
        #         out = out.reshape(out.shape[0], -1)

        return (output, enc_hidden[1][-1]) if return_latent else output