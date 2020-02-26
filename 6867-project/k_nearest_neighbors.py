from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors:
    def __init__(self,n_neighbors):
        self.model=KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
        training_score =self.model.score(X_train,Y_train)
        return training_score
    def score(self,X_val,Y_val):
        validation_score=self.model.score(X_val,Y_val)
        return validation_score

