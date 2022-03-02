from code.stage_5_code.DataLoader_All import Dataset_Loader
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
    node, edge, _ = data_obj.load_raw()
    data_obj.load(node, edge)

    #method_obj = Method_MLP('multi-layer perceptron', '')

    #result_obj = Result_Saver('saver', '')
    #result_obj.result_destination_folder_path = '../../result/stage_1_result/MLP_'
    #result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

   # evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    # print('************ Start ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    # print('************ Finish ************')
    # ------------------------------------------------------