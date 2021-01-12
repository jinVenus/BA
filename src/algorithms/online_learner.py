import pandas as pd
import numpy as np
import torch
from DeepADoTS_master.src.evaluation import Evaluator

# for batch in dataloader.get():
#     predict()
#     if need_update == True:
#         update(buffer)
#         store()
#         clear_buffer()
#     else:
#         append_buffer()

def update(X, buffer, label, updateCount, experiment_folder):
    load_model = torch.load(f'{experiment_folder}/online/phase' + updateCount - 1 + '\\2DimWithOneSudden.th')
    X.interpolate(inplace=True)
    X.bfill(inplace=True)
    data = X.values
    sequences = [data[i:i + 30] for i in (data.shape[0] - 30 + 1)]
    load_model.fit(X, seq=sequences)
    torch.save(load_model, f'{experiment_folder}/online/phase' + updateCount + '\\2DimWithOneSudden.yh')
    score, error, output = load_model.predict(buffer, update=True)
    evaluate = Evaluator()
    t = evaluate.get_optimal_threshold(y_test=label[2000:], score=score)
    print(t)
    return t, score, error, output

def online_learning(model, t, stream_data, sequence_length, batch_size, experiment_folder):
    new_label = np.zeros(sequence_length * batch_size)
    buffer = []
    scores = []
    outputs = []
    errors = []
    threshold = []
    updateCount = 0
    for j in range(stream_data.shape[0] - batch_size * sequence_length + 1):
        if ((j + batch_size * sequence_length) > stream_data.shape[0] - batch_size * sequence_length + 1):
            data = stream_data[j:]
            index = np.arange(0, data.shape[0] - sequence_length, 30, int)
            sequences = [stream_data[i:i + sequence_length] for i in index]

            win_score, win_error, win_output = model.predict(data, seq=sequences, update=False, t=t,
                                                             data=data)
            win_score = pd.DataFrame(win_score)
            win_error = pd.DataFrame(win_error)
            win_output = pd.DataFrame(win_output)

            for id, score in enumerate(win_score):
                if score >= 0.75 * t:
                    buffer.append(stream_data[j:])
                    buffer = np.concatenate(buffer)
                    # if score > t:
                    #     new_label[j+id] = 1
                if len(buffer) >= 3000:
                    updateCount += 1
                    buffer = pd.DataFrame(buffer)
                    for i in range(sequence_length):
                        new_label[1970 + i] = 1
                    t, score, error, output = update(buffer[:2000], buffer, new_label, updateCount, experiment_folder)
                    threshold.append(t)
                    updateCount += 1
                    buffer = []
                    new_label = np.zeros(3000)
                    break
                else:
                    break


            j += batch_size * sequence_length

        else:
            data = stream_data[j:j + batch_size * sequence_length]
            index = np.arange(0, data.shape[0] - sequence_length + 1, sequence_length, int)
            sequences = [data[i:i + sequence_length] for i in index]

            win_score, win_error, win_output = model.predict(data, seq=sequences, update=False, t=t,
                                                             data=data)
            win_score = pd.DataFrame(win_score)
            win_error = pd.DataFrame(win_error)
            win_output = pd.DataFrame(win_output)

            scores.append(win_score)
            outputs.append(win_output)
            errors.append(win_error)

            for id, score in enumerate(win_score):
                if score >= 0.75 * t:
                    buffer.append(stream_data[j:j + batch_size * sequence_length])
                    buffer = np.concatenate(buffer)
                    # if win_score > t:
                    #     new_label[j+id] = 1
                if len(buffer) >= 3000:
                    updateCount += 1
                    buffer = pd.DataFrame(buffer)
                    for i in range(30):
                        new_label[1970 + i] = 1
                    t, score, error, output = update(buffer[:2000], buffer, new_label, updateCount, experiment_folder)
                    threshold.append(t)
                    updateCount += 1
                    buffer = []
                    new_label = np.zeros(3000)
                    break
                else:
                    break
            j += batch_size * sequence_length

    print(threshold)
    print(updateCount)
