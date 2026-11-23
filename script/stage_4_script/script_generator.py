import sys
sys.path.append("../../")

from local_code.stage_4_code.Dataset_Loader_Generator import Dataset_Loader_Generator
from local_code.stage_4_code.Method_Generator import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver

import numpy as np
import torch

# ---- Recurrent Neural Network Text Generation script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    rnn_type = 'RNN' # Can be changed to 'LSTM' and 'GRU'
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader_Generator('Jokes', 'Joke Generation', max_len=64, min_freq=1)
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'

    loaded_data = data_obj.load()
    glove_matrix = data_obj.load_glove_embeddings(
        '../../data/stage_4_data/glove.50d.txt',
        embedding_dim=50
    )

    method_obj = Method_RNN_Generator(f'{rnn_type}_Generator', '', vocab_size=len(loaded_data['word2idx']),
                    embedding_dim=50, hidden_size=256, num_layers=2, dropout = 0.3, rnn_type=rnn_type, pretrained_embeddings=glove_matrix)

    method_obj.epochs = 160
    method_obj.data = loaded_data

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/{rnn_type}_Gen_'
    result_obj.result_destination_file_name = 'generation_result'

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    generated_results = method_obj.run()
    result_obj.data = generated_results
    result_obj.save()

    print('************ Overall Performance ************')
    for prompt, joke in generated_results.items():
        print(f'[{prompt}]: {joke}')
    print('************ Finish ************')
    # ------------------------------------------------------


