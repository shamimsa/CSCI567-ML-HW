import json
import random
import numpy as np


def cluster_points(X, mu):
    """
    Distribute data points into clusters.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - mu: A list of K cluster centers, each elements is a list of 2

    Returns:
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.
    """
    
    clusters = {}

    # you need to fill in your solution here
    K = range(len(mu) + 1)[1:]
    cluster_id_ctr_pairs = list(zip(K,mu)) #assign cluster id's (1:K) to cluster centers
    print(cluster_id_ctr_pairs)
    for item in X:
        dists = {}
        for cluster in cluster_id_ctr_pairs:
            ctr = cluster[1] #cluster center
            cluster_id = cluster[0] #cluster index
            #print(item)
            #print(ctr)
            dist = np.sqrt((item[0] - ctr[0])**2 + (item[1] - ctr[1])**2) 
            #print(dist)
            dists[cluster_id] = dist
        #print(dists)
        item_id = min(dists, key=dists.get) #assigned cluster for the data point
        if item_id in clusters:    
            clusters[item_id].append(item)         
        else:
            clusters[item_id] = [item]
        #print(clusters)
    return clusters


def reevaluate_centers(mu, clusters):
    """
    Update cluster centers.

    Inputs:
    - mu: A list of K cluster centers, each elements is a list of 2
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.

    Returns:
    - newmu: A list of K updated cluster centers, each elements is a list of 2
    """
    #print(clusters)
    newmu = []
    #print(clusters)
    # you need to fill in your solution here
    for cluster_id in clusters.keys():
        print(cluster_id)
        cluster = np.array(clusters[cluster_id])
        #print(cluster)
        new_ctr = list(np.mean(cluster, axis=0))
        #print(new_ctr)
        newmu.append(new_ctr)
    print(newmu)
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    random.seed(100)
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

    return(mu, clusters)


def kmeans_clustering():
    # load data
    with open('hw4_circle.json', 'r') as f:
        data_circle = json.load(f)
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    clusters_all = {}
    for K in [2, 3, 5]:
        key = 'blob_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_blob, K)
        key = 'circle_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_circle, K)

    return mu_all, clusters_all


def main():
    mu_all, clusters_all = kmeans_clustering()

    print('K-means Cluster Centers:')
    for key, value in mu_all.items():
        print('\n%s:'% key)
        print(np.array_str(np.array(value), precision=4))

    with open('kmeans.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'clusters': clusters_all}, f_json)


if __name__ == "__main__":
    main()
