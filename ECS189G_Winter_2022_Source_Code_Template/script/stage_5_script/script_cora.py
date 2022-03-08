from code.stage_5_code.DataLoader_All import Dataset_Loader
from code.stage_5_code.Method_GCN_CORA import Method_GCN_Cora
from code.stage_5_code.Setting_Up_Run import Setting_Up_Run
from code.stage_5_code.Evaluator import Evaluate
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Result_Loader import Result_Loader
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('cora', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/'
    data_obj.dataset_source_file_name = None

    method_obj = Method_GCN_Cora('gcn_cora', '')
    setting_obj = Setting_Up_Run('setting', '')

    setting_obj = Setting_Up_Run('all_metrics', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GCN_cora_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Up_Run('all_metrics', '')

    evaluate_obj = Evaluate('all', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')