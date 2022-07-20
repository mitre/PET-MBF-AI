
import sys
import os
BASE_PATH = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_PATH)

import numpy as np
from codebase.ModelEvalWrapper import ModelEvalWrapper


class LocalizationWrapper:
    """
    Wrapper for to give localization model created from 6 individual binary
    classifiers same interface as sklearn multilabel classifier. Currently
    implemented only for sklearn models
    """     
    def __init__(self, models, feature_cols, db_method=None, db_target=None, db_X_val=None, db_y_val=None):
        """
        models : dictionary of models
            Binary sklearn models for each key of scar_lad, scar_rca, scar_lcx,
            ischemia_lad, ischemia_rca, ischemia_lcx
        
        feature_cols : Dictionary {col_name : col_list}
            col_name is one of:
                'scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad',
                'ischemia_rca', 'ischemia_lcx'
            
            col_list is list of the names of the feature cols that the model 
            was trained on. This must be in the same order as the columns in
            the feature matrix that the model was trained on. This is included
            because the logistic regression used only a subset of the features
            that all other models did, excluding some highly correlated
            features.

        db_method : str
            Optional method used to determine decision boundries. Valid options are
            'youdens' or 'fixed_recall'

        db_target : float
            Optional recall value if db_method is 'fixed_recall'

        db_X_val : pd.DataFrame
            Optional validation feature matrix for setting decision boundry
            based on Youden's index.

        db_y_val : pd.Series
            Optional validation labels for setting decision boundry
            based on Youden's index.

        """
        
        outcomes = ['scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad',
                    'ischemia_rca', 'ischemia_lcx']
        models_wrapped = {}
        for outcome in outcomes:
            lr = ModelEvalWrapper(model=models[outcome],
                          model_name=f"LR {outcome}",
                          model_name_abrv=f"LR_{outcome}",
                          model_type='sklearn',
                          dataset='17_segment',
                          feature_cols=feature_cols[outcome])
            if db_method:
                lr.set_decision_boundries(db_X_val,
                                          db_y_val[outcome],
                                          method=db_method,
                                          target=db_target)
            models_wrapped[outcome] = lr
        self.models = [models_wrapped[outcome] for outcome in outcomes]

    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(X))

        return np.vstack(preds).T

    def predict_proba(self, X):
        probs = []
        for model in self.models:
            probs.append(model.predict_proba(X))
        return probs

