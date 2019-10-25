import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import scipy_read_images as sri
import config

def get_largest_dimensions(X):
    """
    Returns the largets dimensions of all the input features
    Args:
    X: List of feature matrices of different ragas
    Returns:
    (x, y, z) dimensions
    """
    x = y = z = 0
    
    for i in X:
        x = max(x, i.shape[0])
        y = max(y, i.shape[1])
        z = max(z, i.shape[2])

    return x, y, z

def blow_up_matrix(X, x, y, z):
    """
    Increases the dimension of each matrix by padding with zeros
    Args:
    X: List of all matrices
    (x, y, z): largest dimension along each axis

    Returns:
    Feature matrices each having (x, y, z) dimensions
    """

    # memory intensive
    # change later

    A = np.zeros((x, y, z))
    new_X = np.zeros((len(X), x, y, z))

    for k, i in enumerate(X):
        i = np.array(i)
        new_X[k, :i.shape[0], :i.shape[1], :i.shape[2]] = i

    return new_X

def plot_pca():
    """
    Reads images from scipy_read_images file and performs pca

    """

    # TO CHANGE MULTIPLE RAGAS
    X = sri.get_img_data(config.IMG_BEGADA)
    X += sri.get_img_data(config.IMG_VARALI)
    
    x, y, z = get_largest_dimensions(X)

    X = blow_up_matrix(X, x, y, z)


 
    # PCA
    nsamples, x, y, z = X.shape
    new_X = X.reshape((nsamples, x * y * z))
    
    # standardise the data
    new_X = StandardScaler().fit_transform(new_X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(new_X)
    
    # append raga name to every data point


    pc_raga = np.zeros((principal_components.shape[0], principal_components.shape[1] + 1))

    pc_raga[:, [0, 1]] = principal_components
    pc_raga[0:10, 2] = 0 # begada
    pc_raga[10:, 2] = 1 # varali
    #plt.scatter(principal_components[0:10, 0], principal_components[0:10, 1], 'b', principal_components[10:, 0], principal_components[10:, 1], 'r')
    #plt.show()

    color = ["red"] * 10
    blue = ["blue"] * 10
    color += blue
    plt.scatter(pc_raga[:, 0], pc_raga[:, 1], color=color)
    plt.savefig('result_pca.png')

plot_pca()
