from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class TrainModel:
    def __init__(self, data):
        self.train_x = data[0]
        self.train_y = data[1]
        self.test_x = data[2]
        self.test_y = data[3]

    @staticmethod
    def print_grid_search_metrics(gs):
        print('Best Score: ' + str(gs))
        print('Best parameters set: ')
        best_parameters = gs.best_params_
        for param_name in sorted(best_parameters.keys()):
            print(param_name + ':' + str(best_parameters[param_name]))

    def LR(self):
        parameters = {
            'penalty': ('l1', 'l2'),
            'C': (0.01, 0.05, 0.1, 0.2, 1)
        }
        Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=5)
        Grid_LR.fit(self.train_x, self.train_y)
        self.print_grid_search_metrics(Grid_LR)
        best_LR_model = Grid_LR.best_estimator_
        best_LR = best_LR_model.score(self.test_x, self.test_y)
        return best_LR_model

    def kNN(self):
        parameters = {
            'n_neighbors': [1, 3, 5, 7, 9]
        }
        Grid_KNN = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
        Grid_KNN.fit(self.train_x, self.train_y)
        self.print_grid_search_metrics(Grid_KNN)
        best_kNN_model = Grid_KNN.best_estimator_
        best_kNN = best_kNN_model.score(self.test_x, self.test_y)
        return best_kNN_model

    def RF(self):
        parameters = {
            'n_estimators': [60, 80, 100],
            'max_depth': [1, 5, 10]
        }
        Grid_RF = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
        Grid_RF.fit(self.train_x, self.train_y)
        self.print_grid_search_metrics(Grid_RF)
        best_RF_model = Grid_RF.best_estimator_
        best_RF = best_RF_model.score(self.test_x, self.test_y)
        return best_RF_model

    def best_models(self):
        # run logistic regression
        best_LR_model = self.LR()

        # run KNN
        best_kNN_model = self.kNN()

        # run random forest
        best_RF_model = self.RF()

        return best_LR_model, best_kNN_model, best_RF_model


