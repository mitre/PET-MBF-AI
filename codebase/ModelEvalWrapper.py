import sys
import os
BASE_PATH = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_PATH)

import numpy as np
from tensorflow import keras
from codebase import evaluation_utils as eu
import tensorflow.keras.backend as K

class ModelEvalWrapper:
    """
        A wrapper class for sklearn and keras models which will give them
        the same interface for predict, predict_proba, and decision_function.
        This adapts keras interface to appear like that of sklearn models for
        these functions. Parameters X to all methods must be pandas dataframes,
        and self.feature_cols must be a subset (or consist of all of) the 
        columns in X. This class is used to standardize model interface for
        the evaluate.py script and to provide additional information to the
        evaluation script like model name, and model type. 
    """

    def __init__(self, model, model_name, model_name_abrv, model_type, dataset, feature_cols=None, keras_model_path=None, decision_boundries=None):
        """
            Parameters
            ----------

            model : either keras or sklearn model
                A trained model

            model_name : string
                Full name of the model. Will be used in evlauation csvs and
                visualizations

            model_name_abrv : string
                Abreviation for model name. Will be used in evaluation csvs and
                visualizations

            feature_cols : list of strings
                The feature cols that the model was trained on. This must be in
                the same order as the columns in the feature matrix that the
                model was trained on. This is included because the logistic
                regression used only a subset of the features that all other
                models did, excluding some highly correlated features.

            model_type : str
                Either 'sklearn' or 'keras'
            
            dataset : str
                Either '3_vessel', '17_segment', or 'polar_plot'
    
            keras_model_path :string
                Path from BASE_DIR to the HDF5 format keras model saved using
                model.save() 

            decision_boundries : list of floats
                Optional ordered list of decision thresholds to use in prediction.
                There should be one threshold per model output. i.e. if model
                is binary classifier, len(decision_boundries) = 1, and if model is
                multilabel classifier, len(decision_boundries) = # labels. If list
                contains mode than one element, the order of the list 
                corresponds to the order of labels. 
        """

        self.model = model
        self.model_name = model_name
        self.model_name_abrv = model_name_abrv
        self.feature_cols = feature_cols
        self.dataset = dataset
        self.keras_model_path = keras_model_path

        if (model_type == 'sklearn') or (model_type == 'keras') or (model_type  == 'loc_wrapper') :
            self.model_type = model_type
        else:
             raise ValueError("invalid argument: model_type must be either 'sklearn' or 'keras' or 'loc_wrapper'")

        if not decision_boundries:
            if self.model_type == 'loc_wrapper':
                decision_boundries = [model.decision_boundries[0] for model in model.models]
                self.decision_boundries = decision_boundries
            else:
                self.decision_boundries = []
        else:
            self.decision_boundries = decision_boundries


    def set_decision_boundries(self, X_val, y_val, method='youdens', target=None):
        """
        Sets decision boundry for predictions for  binary classification or
        miltilabel classification based on Youden's index.

        Parameters
        ----------
        X_val : pd.DataFrame
            Feature matrix for validation set used to set decision boundry(ies)

        y_val : pd.DataFrame
            Labels for validation set used to set decision boundry(ies)

        Returns
        -------
        None
        """
        decision_boundries = self.calc_decision_boundries(X_val, y_val, method, target)
        self.decision_boundries = decision_boundries


    def calc_decision_boundries(self, X_val, y_val, method='youdens', target=None):
        """
        Returns decision boundry for predictions for  binary classification or
        miltilabel classification based on Youden's index.

        Parameters
        ----------
        X_val : pd.DataFrame
            Feature matrix for validation set used to set decision boundry(ies)

        y_val : pd.DataFrame
            Labels for validation set used to set decision boundry(ies)

        method : str
            Method used to determine decision boundries. Valid options are
            'youdens' or 'fixed_recall'

       target : float
            Target recall value if method is 'fixed_recall' 

        Returns
        -------
        decision_boundries
        """

        assert((method == 'youdens') or (method == 'fixed_recall'))  
        probs = self.predict_proba(X_val)
        y_val = y_val.values
        # Binary case
        if isinstance(probs, np.ndarray):
            probs = [probs]
            y_val = y_val.reshape(-1, 1)

        decision_boundries = []
        for i in range(y_val.shape[1]):
            y_true = y_val[:,i]
            scores = probs[i][:,1]
            if method == 'youdens': 
                dec_boundry = eu.decision_boundry_youdens(y_true,
                                                          scores)
            elif method == 'fixed_recall':
                dec_boundry = eu.decision_boundry_fixed_recall(y_true,
                                                               scores,
                                                               target)
            decision_boundries.append(dec_boundry) 
        return decision_boundries
    

    def predict(self, X):
        BASE_DIR = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'

        try:
            if self.keras_model_path:
                K.clear_session()
                self.model = keras.models.load_model(self.keras_model_path)
        except AttributeError:
            pass

        probs = self.predict_proba(X)
        # Binary case
        if isinstance(probs, np.ndarray):
            probs = [probs]
            binary = True
        else:
            binary = False

        if not self.decision_boundries:
            # Default to 0.5
            decision_boundries = [0.5] * len(probs)
        else:
            decision_boundries = self.decision_boundries

        assert(len(decision_boundries) == len(probs))

        preds = []
        for thresh_i, probs_i in zip(decision_boundries, probs):
            preds_i = np.empty_like(probs_i[:, 1], dtype=int) 
            preds_i[probs_i[:, 1] > thresh_i] = 1
            preds_i[probs_i[:, 1] <= thresh_i] = 0
            preds.append(preds_i)


        preds = np.vstack(preds).T
        if binary:
            preds = preds.reshape(-1,)

        return preds
                
    def predict_proba(self, X):
        BASE_DIR = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'

        try:
            if self.keras_model_path:
                K.clear_session()
                self.model = keras.models.load_model(self.keras_model_path)
        except AttributeError:
            pass

        if self.feature_cols:
            X = X[self.feature_cols]

        if self.model_type != 'keras':
            probs = self.model.predict_proba(X)
        else:
            try:
                X = X.to_numpy()
            except AttributeError:
                pass
            probs_full =  self.model.predict(X)
            if probs_full.shape[1] == 1: 
                prob_of_1 = probs_full.flatten()
                prob_of_0 = 1 - prob_of_1
                probs = np.vstack((prob_of_0, prob_of_1)).T
            else:
                probs_list = []
                for i in range(probs_full.shape[1]):
                    probs_i = probs_full[:,i]
                    prob_of_1 = probs_i.flatten()
                    prob_of_0 = 1 - prob_of_1
                    probs = np.vstack((prob_of_0, prob_of_1)).T
                    probs_list.append(probs)

                probs = probs_list
            
        return probs
