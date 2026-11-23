# Script copied from script_mlp.py

import sys
sys.path.append("../../")

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split

import numpy as np
import torch

# ---- Recurrent Neural Network script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    rnn_type = 'RNN' # Can be changed to 'LSTM' and 'GRU'
    # ------------------------------------------------------


    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('RNN', 'Movie Reviews and Jokes', max_len=256)
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
    data_obj.dataset_source_file_name = 'RNN'

    loaded_data = data_obj.load()
    glove_matrix = data_obj.load_glove_embeddings(
        '../../data/stage_4_data/glove.50d.txt',
        embedding_dim=50
    )

    method_obj = Method_RNN(rnn_type, '', vocab_size=len(glove_matrix), embedding_dim=50,
                            rnn_type=rnn_type, pretrained_embeddings=glove_matrix)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/{rnn_type}_'
    result_obj.result_destination_file_name = 'learning_rate_alter'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(f'{rnn_type} Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------


