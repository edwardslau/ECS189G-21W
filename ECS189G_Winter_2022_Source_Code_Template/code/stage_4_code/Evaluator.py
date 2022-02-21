'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score


class Evaluate(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])
        prec = precision_score(self.data['true_y'], self.data['pred_y'])
        f1 = f1_score(self.data['true_y'], self.data['pred_y'])
        rec = recall_score(self.data['true_y'], self.data['pred_y'])

        results_dict = {"accuracy": acc, "precision": prec, "f1": f1, "recall": rec}

        return results_dict
