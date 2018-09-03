


def CvWrapper(object):

    def __init__(self):
        pass

    def cv(wrapper, folds, X_train, y_train, X_test):
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train):
            fold_x_train, fold_y_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            fold_x_valid, fold_y_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
            wrapper.train(
                n_fold, fold_x_train, fold_y_train, fold_x_valid, fold_y_valid
            )
