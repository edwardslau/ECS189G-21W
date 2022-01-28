from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Up_Run import Setting_Up_Run
#from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    # Loading Training
    data_obj_train = Dataset_Loader('train', '')
    data_obj_train.dataset_source_folder_path = '../../data/stage_2_data/stage_2_data/' # CHANGE NEEDED FOR OTHERS
    data_obj_train.dataset_source_file_name = 'train.csv'

    data_obj_test = Dataset_Loader('test', '')
    data_obj_test.dataset_source_folder_path = '../../data/stage_2_data/stage_2_data/'  # CHANGE NEEDED FOR OTHERS
    data_obj_test.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('mlp', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Up_Run('train_and_test_set', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj_train, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')
    # ------------------------------------------------------


