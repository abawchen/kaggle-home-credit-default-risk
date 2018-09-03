import json
import numpy as np
import os
import pickle

from sklearn.metrics import roc_auc_score

from .utils import timer

class ModelWrapper(object):

    def __init__(
        self, CLF, name, model_folder, feats, drop_feats, params, fit_params):
        self.CLF = CLF
        self.name = name
        self.model_folder = model_folder
        self.feats = feats
        self.drop_feats = drop_feats
        self.params = params
        self.fit_params = fit_params

    def _fit(self, X, y, **fit_params):
        self.clf = self.CLF(**self.params)
        self.clf.fit(X, y, **fit_params)


    def _train(self, n_fold, X_train, y_train, X_valid=None, y_valid=None):
        model_filename = os.path.join(
            self.model_folder, '{}_{}.pickle'.format(self.name, n_fold))

        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                self.clf = pickle.load(f)
        else:
            print('{} not exists, going to train.'.format(model_filename))
            fit_params = self.fit_params.copy()
            if y_valid is not None and 'early_stopping_rounds' in fit_params:
                fit_params['eval_set'] = [(X_valid, y_valid)]
            else:
                fit_params.pop('early_stopping_rounds', None)
            self._fit(X_train, y_train, **fit_params)
            with open(model_filename, 'wb') as f:
                pickle.dump(self.clf, f)

    def _predict(self, filename, X):
        if not os.path.exists(filename):
            preds = self.clf.predict_proba(X)[:, 1]
            np.save(filename, preds)
            return preds
        return np.load(filename)

    def serialize_feats(self):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        with open(os.path.join(self.model_folder, 'feats.json'), 'w') as f:
            json.dump(self.feats, f, indent=2)

        with open(os.path.join(self.model_folder, 'drop_feats.json'), 'w') as f:
            json.dump(self.drop_feats, f, indent=2)

    def serialize_scores(self):
        data = {
            'params': self.clf.get_params(),
            'scores': self.scores
        }
        with open(os.path.join(self.model_folder, 'scores.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def folds_train(self, folds, X_train, y_train, X_test):
        print('feats num:', len(self.feats))
        print('model folder:', self.model_folder)
        self.serialize_feats()
        self.scores = []
        self.oof_preds_df = np.zeros(X_train.shape[0])
        self.test_preds_df = np.zeros(X_test.shape[0])
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train)):
            fold_x_train, fold_y_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            fold_x_valid, fold_y_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
            self._train(
                n_fold, fold_x_train, fold_y_train, fold_x_valid, fold_y_valid
            )
            oof_preds_filename = os.path.join(
                self.model_folder, 'oof_preds_{}.npy'.format(n_fold))
            oof_preds = self._predict(oof_preds_filename, fold_x_valid)
            self.oof_preds_df[valid_idx] = oof_preds
            score = roc_auc_score(fold_y_valid, oof_preds)
            self.scores.append(score)

            test_preds_filename = os.path.join(
                self.model_folder, 'test_preds_{}.npy'.format(n_fold))
            test_preds = self._predict(test_preds_filename, X_test)
            self.test_preds_df += test_preds/folds.n_splits
            yield self.clf, score
