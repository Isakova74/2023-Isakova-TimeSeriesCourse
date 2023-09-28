import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0

    # INSERT YOUR CODE
    square = np.square(ts1 - ts2)
    sum_square = np.sum(square)
    ed_dist = np.sqrt(sum_square)
    
    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    T1_T2 = ts1.dot(ts2) #скалярное произведение

    avg_ts1 = np.mean(ts1) #средее арифметичекое
    avg_ts2 = np.mean(ts2)

    std_ts1 = np.std(ts1) #стандартное отклонение
    std_ts2 = np.std(ts2)

    drob = (T1_T2-avg_ts1*avg_ts2*len(ts1))/(std_ts1*std_ts2*len(ts1))
    norm_ed_dist = abs(2*len(ts1)*(1-drob))**0.5


    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    dtw_dist = 0

    # INSERT YOUR CODE
    M = sum(ts1.shape)
    dist_mat=np.zeros((M,M))
    for i in range(M):
      for j in range(M):
        dist_mat[i,j] = (ts1[i]- ts2[j])**2

    M,M=dist_mat.shape
    D_mat = np.zeros((M+1,M+1))
    for i in range(1,M+1):
        D_mat[i,0]=np.inf
    for i in range(1,M+1):
        D_mat[0,i]=np.inf

    for i in range(1,M+1):
      for j in range(1,M+1):
        D_mat[i][j] = dist_mat[i-1][j-1]+min(D_mat[i-1][j],D_mat[i,j-1],D_mat[i-1][j-1])
    return  D_mat[M][M]

    #return dtw_dist