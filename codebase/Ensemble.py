import numpy as np
import pandas as pd

"""
The Ensemble class is used to ensemble many individual models. The ensemble
predicts a class based on a simple majority vote of its base learners.
Because of this, Ensemble expects an odd number of base models. 
"""

class Ensemble:
    
    def __init__(self, models):
        """
        models : list of ModelEvalWrappers
            The models to include in the ensemble. This should contain
            an odd number of models, as they will predict by majority
            vote.
        """
        if len(models) % 2 == 0:
            raise ValueError("Ensemble expects odd number of models")

        self.models = models

    def _pred_counts(self, X):
        """
            Return the number of base models that classified as class 1 for each
            instance. Currently implemented only for binary or multilabel
            classifiers (not multiclass).
        """
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        
        preds = np.array(preds)
        summed = preds.sum(axis=0)
        return summed
        
        
    def predict(self, X):
        pred_counts = self._pred_counts(X)
        ones_mask = pred_counts > (len(self.models) / 2)
        zeros_mask = ~ones_mask
        preds = pred_counts.copy()
        preds[ones_mask] = 1
        preds[zeros_mask] = 0
        return preds
    
    
    def predict_proba(self, X):
        # Define prob of 1 as the fraction of individual models
        # that predict 1
        pred_counts = self._pred_counts(X)
        probs = pred_counts / len(self.models)

        if len(probs.shape) > 1: 
            probs_list = []
            for i in range(probs.shape[1]):
                probs_array = np.empty_like(probs[:,:2])
                probs_array[:,0] = 1 - probs[:,i]
                probs_array[:,1] = probs[:,i]
                probs_list.append(probs_array)
                
        else:
            probs_array = np.empty_like(probs, shape=(probs.shape[0], 2))
            probs_array[:,0] = 1 - probs
            probs_array[:,1] = probs
            probs_list = probs_array
            
        return probs_list
