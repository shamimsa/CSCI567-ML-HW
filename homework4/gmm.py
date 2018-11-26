import json
import random
import numpy as np


def gmm_clustering(X, K):
    """
    Train GMM with EM for clustering.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - K: A int, the number of total cluster centers

    Returns:
    - mu: A list of all K means in GMM, each elements is a list of 2
    - cov: A list of all K covariance in GMM, each elements is a list of 4
            (note that covariance matrix is symmetric)
    """

    # Initialization:
    pi = []
    mu = []
    cov = []
    for k in range(K):
        pi.append(1.0 / K)
        mu.append(list(np.random.normal(0, 0.5, 2)))
        temp_cov = np.random.normal(0, 0.5, (2, 2))
        temp_cov = np.matmul(temp_cov, np.transpose(temp_cov))
        cov.append(list(temp_cov.reshape(4)))
    #print(pi)
    ### you need to fill in your solution starting here ###
    X = np.array(X) 
    num_data = len(X) #number of data points
    # Run 100 iterations of EM updates
    for t in range(100):
        like = np.zeros((num_data,1))
        post = np.zeros((K, num_data)) #stores posterior for all the classes - each row corresponding to a class k (k=1:K)
        for k in range(K):
            mu_k = np.array(mu[k]).reshape(1,2)
            #print(mu_k.shape)
            #print(X.shape)
            cov_k = np.array(cov[k]).reshape(2,2)
            #print(cov_k.shape)
            pi_k = pi[k]
            logpx_k = []
            for sample in X:
                logpx_samp = - 0.5*(np.dot(sample - mu_k, np.dot(np.linalg.inv(cov_k),np.transpose(sample - mu_k)))) - np.log(2*np.pi) - np.log(np.sqrt(np.linalg.det(cov_k))) + np.log(pi_k)
                #print(logpx_k)
                logpx_k.append(logpx_samp[0][0]) 
            logpx_k = np.array(logpx_k)
            #print(logpx_k.shape)
            #print(logpx_k)
            explog_k = np.exp(logpx_k)
            #print(explog_k.shape)
            #print(post.shape)
            post[k] = explog_k
        like = np.sum(post, axis=0)
        #print(like.shape)
        #print(post.shape)
        post_nrm = post

        mu_new = []
        cov_new = []
        N = 0
        Nk_ls = []
        for k in range(K):
            post_nrm[:][k] = post[:][k] / like #posterior for all the classes
        
        #compute new parameters
            Nk = np.sum(post_nrm[:][k])
            #print(Nk.shape)
            N += Nk
            Nk_ls.append(Nk)
            mu_k_new = np.dot(post_nrm[:][k], X) / Nk
            mu_new.append(list(mu_k_new))
            #print(post_nrm[:][k].shape)
            cov_k_new = np.dot(np.multiply(np.transpose(X - mu_k_new), post_nrm[:][k]), X - mu_k_new) / Nk
            cov_new.append(list(cov_k_new.reshape(4)))

        pi_new = Nk_ls / N
        #update parameters for the next iteration 
        pi = pi_new
        mu = mu_new
        cov = cov_new
    return mu, cov


def main():
    # load data
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    cov_all = {}

    print('GMM clustering')
    for i in range(5):
        np.random.seed(i)
        mu, cov = gmm_clustering(data_blob, K=3)
        mu_all[i] = mu
        cov_all[i] = cov

        print('\nrun' + str(i) + ':')
        print('mean')
        print(np.array_str(np.array(mu), precision=4))
        print('\ncov')
        print(np.array_str(np.array(cov), precision=4))

    with open('gmm.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'cov': cov_all}, f_json)


if __name__ == "__main__":
    main()
