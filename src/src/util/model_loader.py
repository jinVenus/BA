from BA.src.algorithms.lstm_enc_dec_axl import LSTMED
from BA.src.algorithms.gru_enc_dec_axl import GRUED
from BA.src.algorithms.rnn_enc_dec_axl import RNNED
from BA.src.algorithms.dagmm import DAGMM


def ModelLoader(modelname, config_dict):
    if modelname == 'lstm_ae':
        return LSTMED(num_epochs=config_dict["num_epochs"], batch_size=config_dict["batch_size"],
                       hidden_size=config_dict["hidden_size"], sequence_length=config_dict["sequence_length"])

    elif modelname == 'gru_ae':
        return GRUED(num_epochs=config_dict["num_epochs"], batch_size=config_dict["batch_size"],
                      hidden_size=config_dict["hidden_size"], sequence_length=config_dict["sequence_length"])

    elif modelname == 'rnn_ae':
        return RNNED(num_epochs=config_dict["num_epochs"], batch_size=config_dict["batch_size"],
                      hidden_size=config_dict["hidden_size"], sequence_length=config_dict["sequence_length"])
