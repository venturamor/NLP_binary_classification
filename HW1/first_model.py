import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class First_Model():
    def __init__(self):

        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def test(self, x_test):
        '''

        :param x_test:
        :return: (prob, pred)
        '''
        return self.clf.predict_proba(x_test), self.clf.predict(x_test)


    def eval(self, x_eval, y_eval):
        return self.clf.score(x_eval, y_eval)



    # def C_Support_Vector_Classification(X_train, X_test, y_train, y_test, n_splits=5, Classifier='rbf'):
    #     """
    #     k-cross-validation using SVM and classifier building
    #     :param X_train, X_test, Y_train, y_test: train-test splitted data
    #     :param n_splits : number of splitted segmnets (validation and test) from train set
    #     :param Classifier : kernel for SVM
    #     :return: best_svm (classifier)
    #              + plots model performance (confusion matrix) + radar plot
    #     """
    #     verbose = 0
    #     skf = StratifiedKFold(n_splits=n_splits, random_state=15, shuffle=True)
    #     svc = SVC(probability=True)
    #     C = np.array([0.01, 1, 10, 100])
    #     pipe = Pipeline(steps=[('svm', svc)])
    #     if Classifier == 'linear':
    #         svm = GridSearchCV(estimator=pipe,
    #                            param_grid={'svm__kernel': [Classifier], 'svm__C': C},
    #                            scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    #                            cv=skf, refit='roc_auc', verbose=verbose, return_train_score=True)
    #         clf_type = ['linear']
    #     if Classifier == 'rbf' or Classifier == 'poly':
    #         svm = GridSearchCV(estimator=pipe,
    #                            param_grid={'svm__kernel': [Classifier], 'svm__C': C, 'svm__degree': [3],
    #                                        'svm__gamma': ['auto', 'scale']},
    #                            scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    #                            cv=skf, refit='roc_auc', verbose=verbose, return_train_score=True)
    #         clf_type = [Classifier, 'scale']
    #
    #     svm.fit(X_train, y_train)
    #
    #     best_svm = svm.best_estimator_
    #     print(best_svm)
    #
    #     y_pred_test = best_svm.predict(X_test)
    #     y_pred_proba_test = best_svm.predict_proba(X_test)
    #     model_performance(best_svm, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier)
    #
    #     # '''
    #     # Radar function:
    #     plot_radar_svm(svm, clf_type)
    #     plt.grid(False)
    #     # '''
    #
    #     print('C Support Vector Classification -> Done')
    #     return best_svm