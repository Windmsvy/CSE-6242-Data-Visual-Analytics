import csv
import numpy as np  # http://www.numpy.org
import ast 
import random
import time
from math import log


"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of total records
and d is the number of features of each record
Also, y is assumed to be a vector of d labels

XX is similar to X, except that XX also contains the data label for
each record.
"""

"""
This skeleton is provided to help you implement the assignment. It requires
implementing more that just the empty methods listed below. 

So, feel free to add functionalities when needed, but you must keep
the skeleton as it is.
"""

class RandomForest(object):
    class __DecisionTree(object):

        def __init__(self): 
            self.root = self.__DecisionNode()

        # Using __DecisionNode to calculate
        class __DecisionNode(object):

            sFeature = None
            sValue = None
            label = None
            left_child = None
            right_child = None

            # find best split method
            def best_split(self, X, y, average, alreadyTested=[]):

                gain = 0.0
                score = 0.0

                sFeature = None
                sValue = None

                left_child = None
                right_child = None

                # number of column
                nocol=len(X[0])

                # whether y is fulfilled with only 1 element
                if pure(y):
                    self.label = 0
                    return

                #compute the entropy of the node before being split     
                score = entropy(y)

                #split on a feature not yet tested
                for column in range(0,nocol):
                    if column in alreadyTested:
                        pass

                    else:
                        #split on the average value of the feature
                        testValue = average[column] 
                        [X_left,y_left,X_right,y_right] = split(X,y,column,testValue)
                        p = float(len(y_left)/len(y))

                        #calculating information gain
                        testGain = score - p*entropy(y_left) - (1-p)*entropy(y_right)  

                        # split will bring more information entropy
                        if testGain > gain:
                            gain = testGain
                            sFeature = column
                            sValue = testValue

                            Xnew_left = X_left
                            ynew_left = y_left
                            Xnew_right = X_right
                            ynew_right = y_right

                if gain > 0:

                    self.sFeature = sFeature
                    self.sValue = sValue
                    self.left_child = self.__class__()
                    self.right_child = self.__class__()

                    # Grow branches using recursive method
                    alreadyTested = alreadyTested + [sFeature]
                    self.left_child.best_split(Xnew_left, ynew_left, average, alreadyTested)
                    self.right_child.best_split(Xnew_right, ynew_right, average, alreadyTested)

                else:
                    # find majority
                    self.label = np.bincount(y).argmax()
                return

        #learn function
        def learn(self, X, y):
            node = self.root
            average = np.mean(X, axis=0)
            node.best_split(X, y, average)
            pass

        # classify for one record
        def classify(self, record):
            node = self.root
            while node.label == None:
                if record[node.sFeature] >= node.sValue:
                    node = node.left_child
                else:
                    node = node.right_child            
            return node.label
          
    num_trees = 0
    decision_trees = []
    bootstraps_datasets = [] # the bootstrapping dataset for trees
    bootstraps_labels = []   # the true class labels,
                             # corresponding to records in the bootstrapping dataset 

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree() for i in range(num_trees)]
    
    # Generate random array with replacement
    def _bootstrapping(self, XX, n):
        data_sample = [];
        data_label =[];
        t = np.random.randint(len(XX),size=n);
        for number in range(n):
            data_sample.append(XX[t[number]][:-1]);
            data_label.append(XX[t[number]][-1]);
        return data_sample,data_label
        
    # implement
    def bootstrapping(self, XX):
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    # let all decision trees learn
    def fitting(self):
        for i in range(self.num_trees):
            for decision_tree in self.decision_trees:
                X=self.bootstraps_datasets[i]
                y=self.bootstraps_labels[i]
                decision_tree.learn(X,y)
        pass

    # OOB Test
    def voting(self, X):
        y = np.array([], dtype = int)
        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record.tolist() not in dataset:
                    OOB_tree = self.decision_trees[i] 
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)
            if len(counts) == 0:
                # if this record exists in all dataset, using search method to label y with existed value
            	t=0
                while(cmp(record.tolist(),self.bootstraps_datasets[0][t])!=0):
                    t=t+1
                y = np.append(y,self.bootstraps_labels[0][t])
                pass
            else:
                # find majority
                y = np.append(y, np.argmax(counts))
        return y



def main():
    X = list()
    y = list()
    XX = list() # Contains data features and data labels 

    # Note: you must NOT change the general steps taken in this main() function.

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])
            xline = [ast.literal_eval(i) for i in line]
            XX.append(xline[:])

    # Initialize according to your implementation
    forest_size = 10 

    # Initialize a random forest
    randomForest = RandomForest(forest_size)

    # Create the bootstrapping datasets
    randomForest.bootstrapping(XX)

    # Build trees in the forest
    randomForest.fitting()

    # Provide an unbiased error estimation of the random forest 
    # based on out-of-bag (OOB) error estimate.
    # Note that you may need to handle the special case in
    #       which every single record in X has used for training by some 
    #       of the trees in the forest.
    y_truth = np.array(y, dtype = int)
    X = np.array(X, dtype = float)
    y_predicted = randomForest.voting(X)

    #results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
    results = [prediction == truth for prediction, truth in zip(y_predicted, y_truth)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy
    

def entropy(s):
# function for calculate information entropy
    res = 0
    var, counts = np.unique(s, return_counts=True)
    freq = counts.astype('float')/len(s)
    for num in freq:
        if num > 0.0:
            res -= num * np.log2(num)
    return res

# Split function based on variable split_Feature & split_Value(means in this case)
def split(X,y,sFeature,sValue):
    # initialize
    split_function = None

    # Split on binary 
    if isinstance(sValue, int) or isinstance(sValue, float):
        split_function = lambda row : row[sFeature] >= sValue
    else:
        split_function = lambda row : row[sFeature] == sValue

    # Split the rows 
    X_left = np.asarray([row for row in X if split_function(row)])
    X_right = np.asarray([row for row in X if not split_function(row)])

    y_left = np.asarray([y[k] for k,row in enumerate(X) if split_function(row)])
    y_right = np.asarray([y[k] for k,row in enumerate(X) if not split_function(row)])

    return (X_left,y_left,X_right,y_right)

# justify whether a list only contains one unique element
def pure(s):
    return len(np.unique(s))==1

main()
