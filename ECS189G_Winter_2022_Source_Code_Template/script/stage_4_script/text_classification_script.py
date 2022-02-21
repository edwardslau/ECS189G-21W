from code.stage_4_code.Method_RNN_Copy import Method_RNN_Classification
from code.stage_4_code.DataLoader_Class_2 import Dataset_Loader
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Up_Run import Setting_Up_Run
from code.stage_4_code.Evaluator import Evaluate
import torch
import numpy as np

if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('IMDB Dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
    data_obj.dataset_source_file_name = None

    method_obj = Method_RNN_Classification('RNN', '', vocab_size=10000 + 2)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/text_classification_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Up_Run('all_metrics', '')

    evaluate_obj = Evaluate('all', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')
    # ------------------------------------------------------