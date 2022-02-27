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
        context, targets, vocab_w_i, vocab_i_w, jokes, jokes_len = self.dataset.load()

        # run MethodModule
        self.method.data = {'X': context, 'y': targets, 'w_i' : vocab_w_i, 'i_w' : vocab_i_w,
                            'jokes' : jokes, "jokes_len" : jokes_len}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()