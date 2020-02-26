# Class for training our model with the SVM
# TODO: Make it inherit from some model base class?

import sklearn

class SVM:
    def __init__(self):
        self.model = sklearn.svm.LinearSVC()
   
    def train(self,X_train,Y_train):
        self.model.fit(X_train, Y_train)
        train_score = self.model.score(X_train, Y_train)
        return train_score

    def score(self,X_val,Y_val):
        valid_score = self.model.score(X_val,Y_val)
        return valid_score    

