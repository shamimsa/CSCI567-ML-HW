import json
import numpy as np


###### Q5.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here
    N = X.shape[0]
    loss_term = np.ones((N,)) - np.multiply(np.transpose(y), np.dot(np.transpose(w), np.transpose(X)))
    loss_term_pos = loss_term[loss_term > 0]
    obj_value = (lamb/2) * np.dot(np.transpose(w), w) + (1/N) * np.sum(loss_term_pos)
    obj_value = obj_value[0][0]
    return obj_value


###### Q5.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: Aolist of num_train elements, where each element is a list of D-dimensional features.

    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []
    
    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch

        # you need to fill in your solution here
        At_plus = []
        sum_term = np.zeros((1,D))
        for idx in A_t:
            if ytrain[idx] * np.dot(np.transpose(w), Xtrain[idx][:]) < 1:
                At_plus.append(idx)
                sum_term += ytrain[idx] * Xtrain[idx][:] 
        At_plus = np.array(At_plus)
        sum_term = np.transpose(sum_term)
        eta_t = 1 / (lamb * iter)
        #print(w.shape)
        #print(sum_term.shape)
        w_temp = w * (1 - eta_t * lamb) + sum_term * (eta_t / k)
        #print(w_temp.shape)
        if (1 / lamb ** (1 / 2)) / np.dot(np.transpose(w_temp), w_temp) ** (1 / 2) < 1:
            w = ((1 / lamb ** (1 / 2)) / np.dot(np.transpose(w_temp), w_temp) ** (1 / 2)) * w_temp
        else:
            w = w_temp
        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))               
    return w, train_obj


###### Q5.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    pred = np.dot(np.transpose(w), np.transpose(Xtest))
    y_pred = []
    #print(pred) 
    for pred_val in pred[0]:
        if pred_val >= t:
            y_pred.append(1)
        else:
            y_pred.append(-1) 
    y_pred = np.array(y_pred)
    true_ct = 0
    for i in range(len(Xtest)):
        if y_pred[i] == ytest[i]:
            true_ct += 1
    test_acc = true_ct / len(Xtest)    
    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
