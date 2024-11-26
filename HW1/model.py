import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        
        
        x_train_tmp = [tup[0] for tup in train_data]
        self.x_train = np.array(x_train_tmp).reshape(len(x_train_tmp), -1)
        self.y_train = [tup[1] for tup in train_data]
            
        x_test_tmp = [tup[0] for tup in test_data]
        self.x_test = np.array(x_test_tmp).reshape(len(x_test_tmp), -1)
        self.y_test = [tup[1] for tup in test_data]
        
        # raise NotImplementedError("To be implemented")
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        
        if model_name == "RF":
            rfc = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state =50, min_samples_leaf = 10)
            return rfc
        
        
        elif model_name == "KNN":
            knn5 = KNeighborsClassifier(n_neighbors = 5)
            return knn5
            
            
            
        elif model_name == "AB":
            adaboost = AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)
            return adaboost
            
        # raise NotImplementedError("To be implemented")
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        
        self.model.fit(self.x_train, self.y_train)
        
        # raise NotImplementedError("To be implemented")
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input)[0]
        

