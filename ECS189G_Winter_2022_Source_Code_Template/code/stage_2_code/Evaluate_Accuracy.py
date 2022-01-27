'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        #if self.evaluate_name == 'precision':
            #return precision_score(self.data['true_y'], self.data['pred_y'])
        #if self.evaluate_name == 'recall':
            #return recall_score(self.data['true_y'], self.data['pred_y'])
        #if self.evaluate_name == 'f1':
           # return f1_score(self.data['true_y'], self.data['pred_y'])

        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])
        prec = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        rec = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')

        results_dict = {"accuracy": acc, "precision": prec, "f1": f1, "recall": rec}

        return results_dict
