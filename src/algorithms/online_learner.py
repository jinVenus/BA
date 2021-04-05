import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from DeepADoTS_master.src.evaluation import Evaluator
import matplotlib.pyplot as plt
from BA.src.algorithms.ks_test import kstest, ksTest
from BA.src.util.DataPreprocessing import splitWindow
from BA.src.algorithms.action import active

# for batch in dataloader.get():
#     predict()
#     if need_update == True:
#         update(buffer)
#         store()
#         clear_buffer()
#     else:
#         append_buffer()

def update(stream_data, buffer, label, updateCount, experiment_folder, sequence_length, batch_size, index,
           streamSequences):
    load_model = torch.load('E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
                            '\\phase%d' % (updateCount - 1) + '_2DimWithOneSudden.th')
    # buffer = pd.DataFrame(buffer)
    # # buffer.interpolate(inplace=True)
    # # buffer.bfill(inplace=True)
    # buffer = buffer.values
    # stream_data = pd.DataFrame(stream_data)
    cal = []
    nbuffer = []
    abuffer = []
    stream_data.interpolate(inplace=True)
    stream_data.bfill(inplace=True)
    stream_data = stream_data.values
    # sequences = [data[i:i + 30] for i in (data.shape[0] - 30 + 1)]

    sum = 0
    mu = 0
    sigma = 0

    for bn in buffer:
        for bs in bn:
            sum += bs
            cal.append(bs)

    cal = np.array(cal)

    mu = sum / (sequence_length + len(buffer))
    sigma = np.std(cal, ddof=1)

    threshold = mu + 3 * sigma
    # for bn in buffer:
    #     if len(bn >= threshold) > 0:
    #         abuffer.append(bn)
    #     else:
    #         nbuffer.append(bn)

    load_model.fit(stream_data, seq=buffer)

    # mu = pd.DataFrame(eval(mu))
    # sigma = pd.DataFrame(eval(sigma))
    # mu.to_csv(
    #     'E:\\ML\\BA\\results\\exp_2021_01_07\\lstm_dagmm\\win_30_hs_80_lr_0.01_agots_sudden\\online\\phase%d' % (
    #         updateCount) + 'parameters\\mu.csv',
    #     index=False, sep=',')
    # sigma.to_csv(
    #     'E:\\ML\\BA\\results\\exp_2021_01_07\\lstm_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online\\phase%d' % (
    #         updateCount) + 'parameters\\sigma.csv',
    #     index=False, sep=',')

    torch.save(load_model, 'E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
                           '\\phase%d' % (updateCount) + '_2DimWithOneSudden.th')
    # score, error, output = load_model.predict(stream_data, seq=streamSequences[index:index + 200], update=False,
    #                                           data=stream_data)
    #
    # if len(score) < len(label):
    #     label = label[-len(score):]
    #
    # evaluate = Evaluator()
    # t = evaluate.get_optimal_threshold(y_test=label, score=score)
    # print(t)


def online_learning(model, t_1, streamSequences, stream_data, sequence_length, batch_size, experiment_folder, label):
    names = {}
    new_label = np.zeros(200)
    buffer = []
    abuffer = []
    scores = []
    save = []
    threshold = []
    updateCount = 0
    begin = 0
    ind = 0

    stream_data = pd.DataFrame(stream_data)

    for i in stream_data.columns.values:
        act = []
        count = 0

        while begin == 0 or ind != 0:
            begin = 1
            test = stream_data[[i]]
            test = test[ind:]
            ind, st = ksTest(test)

            if ind != 0:
                act = active(act, ind)
                count += 1
                if len(act) > 1 and act[count-1][0] - act[count-2][0] <= 2000 and count > 0:
                    ind = ind[0]
                else:
                    if ind[0] >= 50:

                        start = ind[0] / 250
                        start = int(start)
                        ind = ind[0]
                    else:
                        start = 0
                        ind = 0

                    buffer = streamSequences[start:start + 20]
                    save = streamSequences[start:start + 20]
                    save = np.concatenate(save)
                    # max_index = np.argmin(buffer, axis=1)
                    # print(max_index)
                    save = pd.DataFrame(save)
                    save.to_csv('E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
                                '\\phase%d' % (updateCount) + 'buffer.csv', index=False, header=False)
                    save = []
                    #
                    # for i in range(30):
                    #     new_label[170 + i] = 1
                    updateCount += 1
                    update(stream_data, buffer, new_label, updateCount, experiment_folder,
                           sequence_length, batch_size, start, streamSequences)
                    # threshold.append(new_t)
                    # scores.append(score)
                    buffer = []

            if count >= 3 and ind == 0:
                if updateCount <= 1:
                    if act[-1][0] >= 50:

                        start = act[-1][0] / 250
                        start = int(start)
                        ind = act[-1][0]
                    else:
                        start = 0
                        ind = 0
                buffer = streamSequences[start:start + 20]
                save = streamSequences[start:start + 20]
                save = np.concatenate(save)
                # max_index = np.argmin(buffer, axis=1)
                # print(max_index)
                save = pd.DataFrame(save)
                save.to_csv(
                    'E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
                    '\\phase%d' % (updateCount) + 'buffer.csv', index=False, header=False)
                save = []
                updateCount += 1
                update(stream_data, buffer, new_label, updateCount, experiment_folder,
                       sequence_length, batch_size, start, streamSequences)

                break

        begin = 0



    stream_data.interpolate(inplace=True)
    stream_data.bfill(inplace=True)
    stream_data = stream_data.values

    updateCount = 1

    for i in range(updateCount + 1):

        names['model_' + str(i)] = torch.load(
            'E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
            '\\phase%d' % i + '_2DimWithOneSudden.th')

    # for model in names:
    #
    #     score, error, output = model.predict(stream_data[:20100], update=True)
    #     scores.append(score)

    model_1 = torch.load(
        'E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
        '\\phase%d' % (updateCount-1) + '_2DimWithOneSudden.th')

    score, error, output = model_1.predict(stream_data[:20100], update=True)

    scores.append(score)
    #
    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-3) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[30300:31300], update=True)
    #
    # scores.append(score)

    model_3 = torch.load(
        'E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\online'
        '\\phase%d' % (updateCount) + '_2DimWithOneSudden.th')

    score, error, output = model_3.predict(stream_data[20100:], update=True)

    scores.append(score)

    # model_4 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-1) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_4.predict(stream_data[32300:33300], update=True)
    #
    # scores.append(score)
    #
    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[33300:], update=True)
    #
    # scores.append(score)

    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[34200:], update=True)
    #
    # scores.append(score)

    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-4) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[35300:36300], update=True)
    #
    # scores.append(score)
    #
    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-3) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[36300:37300], update=True)
    #
    # scores.append(score)
    #
    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-2) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[37300:38300], update=True)
    #
    # scores.append(score)
    #
    # model_2 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount-1) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_2.predict(stream_data[38300:39300], update=True)
    #
    # scores.append(score)
    #
    # model_3 = torch.load(
    #     'E:\\ML\\BA\\results\\exp_2021_03_28\\lstm_ae\\win_30_hs_80_lr_0.01_agots_incremental\\online'
    #     '\\phase%d' % (updateCount) + '_2DWithOneIncremental.th')
    #
    # score, error, output = model_3.predict(stream_data[39300:], update=True)
    #
    # scores.append(score)

    # score = np.concatenate(score)
    # lattice = np.full((sequence_length, stream_data[4000:].shape[0]), np.nan)
    # for i, s in enumerate(score):
    #     lattice[i % sequence_length, i:i + sequence_length] = s
    # score = np.nanmean(lattice, axis=0)
    # evaluate = Evaluator()
    # new_t = evaluate.get_optimal_threshold(y_test=label[9999:], score=score)

    scores = np.concatenate(scores)

    scores = pd.DataFrame(scores)
    scores.to_csv('E:\\ML\\BA\\results\\exp_2021_03_28\\rnn_ae\\win_30_hs_80_lr_0.01_agots_sudden\\scores.csv',
                  index=False, header=False)
    # lattice = np.full((sequence_length, stream_data.shape[0]), np.nan)
    # for i, score in enumerate(scores):
    #     lattice[i % sequence_length, i:i + sequence_length] = score
    # scores = np.nanmean(lattice, axis=0)
    scores = np.array(scores)

    scores[-1] = scores[-2]

    result_test = [0] * (len(scores))
    for i in range(len(scores)):
        if scores[i] >= t_1:
            result_test[i] = 1

    # scores = np.concatenate(scores)
    print(threshold)
    print(updateCount)

    print("data : 6000-20000")
    auc = roc_auc_score(label[30000:], scores)
    # recall = recall_score(label[5999:], result_test)
    con = confusion_matrix(label[30000:], result_test)
    print("Auc : " + str(auc))
    # print("recall : " + str(recall))
    print("confusion_matrix : " + str(con))

    plt.plot(scores)
    plt.show()

    plt.plot(scores[29000:35000])
    plt.show()

    # plt.plot(scores[39000:41000])
    # plt.show()
