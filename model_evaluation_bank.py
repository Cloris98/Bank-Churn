from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt


class ModelEval:
    def __init__(self, lr, knn, rf):
        self.lr = lr
        self.knn = knn
        self.rf = rf

    # calculate accuracy, precision, recall and F1-score, [[tn, tp], []]
    def cal_evaluation(self, classifier, cm):
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        accuracy = (tp + tn) / (tp + tn + fp + fn + 0.0)
        precision = tp / (tp + fp + 0.0)
        recall = tp / (tp + fn + 0.0)
        f1_score = 2 / (1/recall + 1/precision)
        print(classifier)
        print('Accuracy is: ' + str(accuracy))
        print('precision is: ' + str(precision))
        print('recall is: ' + str(recall))
        print('f1-score is: ' + str(f1_score))
        print()

    def draw_confusion_matrix(self, confusion_matricies):
        class_name = ['Not', 'Churn']
        for cm in confusion_matricies:
            classifier, cm = cm[0], cm[1]
            self.cal_evaluation(classifier, cm)

    def ROC_AUC(self,test_x, test_y, best_model):
        y_predict = best_model.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, y_predict)
        # AUC score
        return metrics.auc(fpr, tpr)

    def eval(self, test_x, test_y, model_1, model_2, model_3):
        confusion_matricies = [
            (str(model_1), confusion_matrix(test_y, model_1.predict(test_x))),
            (str(model_2), confusion_matrix(test_y, model_2.predict(test_x))),
            (str(model_3), confusion_matrix(test_y, model_3.predict(test_x)))
        ]
        self.draw_confusion_matrix(confusion_matricies)

        # random forest AUC score
        self.ROC_AUC(test_x, test_y, model_1)
        self.ROC_AUC(test_x, test_y, model_2)
        self.ROC_AUC(test_x, test_y, model_3)