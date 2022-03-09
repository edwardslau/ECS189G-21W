'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np


class Setting_Up_Run(setting):

    def load_run_save_evaluate(self):
        """
        This code loads the dataset, trains the model, and reports metrics on the efficacy of the model.

        :return: None
        """
        # load dataset
        node, edge, class_dict = self.dataset.load_raw()
        loaded_data = self.dataset.load(node, edge)

        # run MethodModule
        self.method.data = loaded_data
        self.method.class_dict = class_dict
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
        #
        self.evaluate.data = learned_result

        # print(self.evaluate.evaluate())