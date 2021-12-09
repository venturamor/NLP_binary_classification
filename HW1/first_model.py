import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, plot_roc_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class First_Model():
    def __init__(self, refit='f1', kernel='rbf', n_splits=3):
        '''
        :param refit:
        :param kernel:
        :param n_splits:
        '''
        # self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        # Pipeline(steps=[('scaler', StandardScaler()),
        #                 ('svm', SVC(C=100.0, gamma='auto', probability=True))])

        verbose = 2
        skf = StratifiedKFold(n_splits=n_splits, random_state=15, shuffle=True)
        svc = SVC(probability=True)
        C = np.array([1.25])  # 1, 1.5, 10, 100])
        # C = np.array([0.01])
        pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svm', svc)])
        # pipe = Pipeline(steps=[StandardScaler(), ('svm', svc)])
        if kernel == 'linear':
            # self.clf = svm
            self.clf = GridSearchCV(estimator=pipe,
                                    param_grid={'svm__kernel': [kernel], 'svm__C': C},
                                    scoring=['f1'],  # , 'accuracy', 'precision', 'recall', 'roc_auc'],
                                    cv=skf, refit=refit, verbose=verbose, return_train_score=True)
            clf_type = ['linear']
        if kernel == 'rbf' or kernel == 'poly':
            # kernels = ['rbf', 'poly']
            self.clf = GridSearchCV(estimator=pipe,
                                    param_grid={'svm__kernel': [kernel], 'svm__C': C, 'svm__degree': [3],
                                                'svm__gamma': ['auto']},  # , 'scale'
                                    scoring=['f1'],  # , 'accuracy', 'precision', 'recall', 'roc_auc'],
                                    cv=skf, refit=refit, verbose=verbose, return_train_score=True)
            clf_type = [kernel, 'scale']

        self.best_clf = None

    def train(self, x_train, y_train):
        '''

        :param x_train:
        :param y_train:
        :return:
        '''
        self.clf.fit(x_train, y_train)
        self.best_clf = self.clf.best_estimator_
        print(self.best_clf)
        print('train evaluation - f1 score: ', self.eval(x_eval=x_train, y_eval=y_train))
        return self.best_clf


    def test(self, x_test):
        '''

        :param x_test:
        :return: (prob, pred)
        '''
        # return self.clf.predict_proba(x_test), self.clf.predict(x_test)
        y_pred = self.best_clf.predict(x_test)
        y_prob = self.best_clf.predict_proba(x_test)

        return y_prob, y_pred


    def eval(self, x_eval, y_eval):
        '''
        :param x_eval:
        :param y_eval:
        :return: score (by refit) of model over data
        '''
        # return self.clf.score(x_eval, y_eval)
        # return self.best_clf.score(x_eval, y_eval)
        score = self.best_clf.score(x_eval, y_eval)
        print(' eval score: ', score)
        return score

    def model_performance(self, X_test, y_test, y_pred_test, y_pred_proba_test, clf):
        """
        confusion matrix and statistics of the classifier
        :param clf : trained classifier
        :param y_pred_test, y_pred_proba_test : prediction set
        :param  X_test, y_test: test set
        :return: confusion matrix plot and statistics printed
        """
        print(classification_report(y_test, y_pred_test))

        calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
        calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
        calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
        calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

        # plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
        # plt.show()

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred_test, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.show()
        #

        TN = calc_TN(y_test, y_pred_test)
        FP = calc_FP(y_test, y_pred_test)
        FN = calc_FN(y_test, y_pred_test)
        TP = calc_TP(y_test, y_pred_test)
        Se = TP / (TP + FN)  # recall
        Sp = TN / (TN + FP)
        PPV = TP / (TP + FP)  # percision
        NPV = TN / (TN + FN)
        Acc = (TP + TN) / (TP + TN + FP + FN)
        F1 = (2 * Se * PPV) / (Se + PPV)

        print(clf, ':')

        print('Sensitivity is {:.2f} \n'
              'Specificity is {:.2f} \n'
              'PPV is {:.2f} \n'
              'NPV is {:.2f} \n'
              'Accuracy is {:.2f} \n'
              'F1 is {:.2f} '.format(Se, Sp, PPV, NPV, Acc, F1))

        print('AUROC is {:.3f}'.format(roc_auc_score(y_test, y_pred_proba_test[:, 1])))
