import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import matplotlib.pyplot as plt
from operator import itemgetter

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = 0
        self.count_positive = 0
        self.count_negative = 0
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0
        self.P_negative = 0
        self.deno_pos = 0
        self.deno_neg = 0
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        X1 = X.todense()
        
        #We initialize two lists with zero values for each word
        self.count_positive = []
        self.count_negative = []
        for n in range(0, X1.shape[1]):
            self.count_positive.append(0)
            self.count_negative.append(0)
        
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        
        #We fill the lists with number of times that each word appears for each class
        for n in range(0, X1.shape[0]):
            if Y[n] == 1.0:
                self.num_positive_reviews += 1
                self.count_positive = [sum(x) for x in zip(self.count_positive, X1[n].tolist()[0])]
            if Y[n] == -1.0:
                self.num_negative_reviews += 1
                self.count_negative = [sum(x) for x in zip(self.count_negative, X1[n].tolist()[0])]
        
        #We count the total number of words in both classes
        self.total_positive_words = 0
        for n in range(0, len(self.count_positive)):
            self.total_positive_words = self.total_positive_words + self.count_positive[n]
        
        self.total_negative_words = 0
        for n in range(0, len(self.count_negative)):
            self.total_negative_words = self.total_negative_words + self.count_negative[n]
            
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        self.P_positive = 0
        self.P_negative = 0
        
        #Using the formula of the slides
        score_pos = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
        score_neg = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
        
        pred_labels = []
        
        sh = X.shape[0]
        
        #For each document
        for i in range(sh):
            #Take indexes of nonzero values
            z = X[i].nonzero()
            for j in range(len(z[0])):
                index = z[1][j]
                # Look at each feature
                a = self.count_positive[index]
                if index > len(self.count_positive)-1:
                    a = 0
                score_pos = score_pos + X[i, index]*(log((a+(self.ALPHA))/(self.total_positive_words + (self.ALPHA*np.sum(X[i])))))
                
                b = self.count_negative[index]
                if index > len(self.count_negative)-1:
                    b = 0
                score_neg = score_neg + X[i, index]*(log((b+(self.ALPHA))/(self.total_negative_words + (self.ALPHA*np.sum(X[i])))))

            if score_pos > score_neg:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)
            
            #Variables update
            score_pos = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
            score_neg = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
        
        #print(pred_labels)
        return pred_labels

    def LogSum(self, logx, logy): 
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):
        pred_labels = []
        
        part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
        part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
        
        if len(indexes) > test.X.shape[0]:
            indexes = []
            for i in range (0, test.X.shape[0]):
                indexes.append(i)
        
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            predicted_prob_positive = 0
            predicted_prob_negative = 0
            numerator_pos = 0
            numerator_neg = 0
            denominator = 0
            part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
            part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
            
            z = test.X[i].nonzero()
            for j in range(len(z[0])):
                index = z[1][j]
                # Look at each feature
                a = self.count_positive[index]
                if index > len(self.count_positive)-1:
                    a = 0
                part_1 = part_1 + test.X[i,index]*(log((a+(self.ALPHA))/(self.total_positive_words + (self.ALPHA*np.sum(test.X[i])))))
                
                b = self.count_negative[index]
                if index > len(self.count_negative)-1:
                    b = 0
                part_2 = part_2 + test.X[i,index]*(log((b+(self.ALPHA))/(self.total_negative_words + (self.ALPHA*np.sum(test.X[i])))))
                
            numerator_pos = part_1
            numerator_neg = part_2
            denominator = self.LogSum(numerator_pos, numerator_neg)
            
            predicted_prob_positive = exp(numerator_pos - denominator)
            predicted_prob_negative = exp(numerator_neg - denominator)
            
            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
                pred_labels.append(predicted_label)
            else:
                predicted_label = -1.0
                pred_labels.append(predicted_label)
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
            
            #Variables update
            part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
            part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
            
        #print(pred_labels)
        return pred_labels
    
    # Predict the probability of each indexed review in sparse matrix text
    # of being positive given a threshold
    # Prints results
    def PredictProbThreshold(self, test, indexes, probThres):
        pred_labels = []
        
        part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
        part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
        
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            predicted_prob_positive = 0
            predicted_prob_negative = 0
            numerator_pos = 0
            numerator_neg = 0
            denominator = 0
            part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
            part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
            
            z = test.X[i].nonzero()
            for j in range(len(z[0])):
                index = z[1][j]
                # Look at each feature
                a = self.count_positive[index]
                if index > len(self.count_positive)-1:
                    a = 0
                part_1 = part_1 + test.X[i,index]*(log((a+(self.ALPHA))/(self.total_positive_words + (self.ALPHA*np.sum(test.X[i])))))
                
                b = self.count_negative[index]
                if index > len(self.count_negative)-1:
                    b = 0
                part_2 = part_2 + test.X[i, index]*(log((b+(self.ALPHA))/(self.total_negative_words + (self.ALPHA*np.sum(test.X[i])))))
                
            numerator_pos = part_1
            numerator_neg = part_2
            denominator = self.LogSum(numerator_pos, numerator_neg)
            
            predicted_prob_positive = exp(numerator_pos - denominator)
            predicted_prob_negative = exp(numerator_neg - denominator)
            
            if (predicted_prob_positive > predicted_prob_negative and predicted_prob_positive >= probThres):
                predicted_label = 1.0
                pred_labels.append(predicted_label)
            else:
                predicted_label = -1.0
                pred_labels.append(predicted_label)
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative)
            
            #Variables update
            part_1 = log(self.total_positive_words/(self.total_positive_words + self.total_negative_words))
            part_2 = log(self.total_negative_words/(self.total_positive_words + self.total_negative_words))
          
        #print(test.Y.tolist())
        #print(pred_labels)
        return pred_labels
    
    # Evaluate precision on a given class 
    # Precision = TP / (TP+FP)
    def EvalPrecision(self, test, threshold):
        indexes = []
        for i in range(0, test.X.shape[0]):
            indexes.append(i)
        Y_pred = self.PredictProbThreshold(test, indexes, threshold)
        TP = 0
        FP = 0

        print("Threshold: " + str(threshold))

        for i in range(0, test.Y.shape[0]):
            if Y_pred[i] == 1.0:
                if float(test.Y[i]) == 1.0:
                    TP += 1
                else:
                    FP +=1
        if (TP+FP) != 0:
            precision = TP/(TP+FP)
        else:
            precision = 0.0
        
        return precision
    
    # Evaluate recall on a given class 
    # Recall = TP / (TP+FN)
    def EvalRecall(self, test, threshold):
        indexes = []
        for i in range(0, test.X.shape[0]):
            indexes.append(i)
        Y_pred = self.PredictProbThreshold(test, indexes, threshold)
        TP = 0
        FN = 0
        
        for i in range(0, test.Y.shape[0]):
            if test.Y[i] == 1.0:
                if Y_pred[i] == 1.0:
                    TP += 1
                else:
                    FN +=1
        if (TP+FN) != 0:
            recall = TP/(TP+FN)
        else:
            recall = 0.0
        
        return recall
    
    # Evaluate performance on test data 
    def EvalLabels(self, test):
        Y_pred = self.PredictLabel(test.X)

        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    
    # Evaluate performance on test data 
    def EvalProbabilities(self, test, indexes):
        Y_pred = self.PredictProb(test, indexes)
        
        ev = Eval(Y_pred, test.Y[indexes])
        return ev.Accuracy()
    
    # Prints out the n most positive and n most negative words in the vocabulary  
    def Features(self, data, n):
        X = data.X
        positive = []
        negative = []
        for i in range(0, X.shape[1]):
            positive.append([data.vocab.GetWord(i), self.count_positive[i]/self.total_positive_words])
            negative.append([data.vocab.GetWord(i), self.count_negative[i]/self.total_negative_words])
        positive_ordered = sorted(positive, key=itemgetter(1), reverse = True)[0:n]
        negative_ordered = sorted(negative, key=itemgetter(1), reverse = True)[0:n]
        
        print("20 positive words with a higher probability (word, probability): ")
        print(positive_ordered)
        print("20 negative words with a higher probability (word, probability): ")
        print(negative_ordered)

if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)

    #First part of the assignment: Classification and Evaluation
    print("1. CLASSIFICATION AND EVALUATION")
    ALPHAs = [0.1, 0.5, 1.0, 5.0, 10.0]
    for i in range(0, len(ALPHAs)):
        print("Computing Parameters")
        print("ALPHA = " + str(ALPHAs[i]))
        if ALPHAs[i] == 1.0:
            nb = NaiveBayes(traindata, ALPHAs[i])
            print("Evaluating")
            print("Test Accuracy: ", nb.EvalLabels(testdata))
        else:
            nb1 = NaiveBayes(traindata, ALPHAs[i])
            print("Evaluating")
            print("Test Accuracy: ", nb1.EvalLabels(testdata))
    

    nb = NaiveBayes(traindata, 1.0)



    #Second part of the assignment: Probability Prediction
    print("2. PROBABILITY PREDICTION")
    indexes = [0, 1, 2, 3, 4, 5, 6]
    print("Test Accuracy: ", nb.EvalProbabilities(testdata, indexes))

    #Third part of the assignment: Precision and Recall
    print("3. PRECISION AND RECALL")
    precisions = []
    recalls = []
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("Thresholds: ")
    print(threshold)
    for i in range(0, len(threshold)):
        precisions.append(nb.EvalPrecision(testdata, threshold[i]))
        recalls.append(nb.EvalRecall(testdata, threshold[i]))
    print("Precisions: ")
    print(precisions)
    print("Recalls: ")
    print(recalls)
    #plt.plot(threshold, precisions)
    #plt.plot(threshold, recalls)
    #plt.show()
    
    #Fourth part of the assignment: Precision and Recall
    print("4. FEATURES")
    nb.Features(traindata, 20)

