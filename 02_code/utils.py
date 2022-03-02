import pandas as pd
import numpy as np

# define function for gradient descent algorithm with least square regression
def gradient_descent(X, y, step_size, precision=1e-6):
    """ 
    input: 
        X: numpy array (n,m), 
        y: numpy array (n,), 
        step_size: int, 
        precision: float (0...1)
    returns:
        betas_new (numpy array (m,))
    """
    # initialize, so that the first iteration does not stop due to precision
    betas_new = np.zeros(X.shape[1])
    betas_hist = np.ones_like(betas_new)
    counter = 0
    while np.linalg.norm(betas_new - betas_hist) / np.linalg.norm(betas_hist) > precision:
        betas_hist = betas_new

        prediction = X.dot(betas_new)
        betas_new = betas_new + 2* step_size * (y - prediction).T.dot(X)
        
        counter += 1
        
    return betas_new, counter

if __name__ == "__main__":
    df_X = pd.read_csv("../01_data/syn_X.csv", header=None)
    df_y = pd.read_csv("../01_data/syn_y.csv", header=None)

    # prepare the vectors/ matrices 
    X = df_X.to_numpy()
    X = np.concatenate((np.ones_like(X[:,:1]), X), axis=1)
    y = np.squeeze(df_y.to_numpy())
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    alphas = [0.004, 0.0041, 0.0042, 0.0043, 0.00435, 0.0044, 0.0045]
    print("betas: {}" .format(betas))
    for alpha in alphas:
        print("alpha = {}: {}, {}" .format(alpha, gradient_descent(X, y, alpha)[0],gradient_descent(X, y, alpha)[1]))
    # result: alpha = 0.00435