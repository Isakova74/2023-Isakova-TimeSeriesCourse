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
    
   # ed_dist = 0

    # INSERT YOUR CODE
    square=np.square(ts1-ts2)######333333333**2
    sum_sqr=np.sum(square)
    ed_dist = np.sqrt(sum_sqr)

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


def DTW_distance(ts1, ts2, r=None): ##r=0.03):
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
    r=0.01
    r = int(np.round(r*len(ts1)))

    N, M = len(ts1), len(ts2)

    dist_mat=np.zeros((N, M))
    for i in range(N):
      start_j = max(1, i-r)-1
      end_j = min(len(ts1),i+r)
      for j in range(start_j, end_j):
        dist_mat[i,j] = (ts1[i] - ts2[j])**2

    D_mat = np.zeros((N+1, M+1))
    for i in range(1,N+1):
        D_mat[i,0]=np.inf
    for i in range(1,M+1):
        D_mat[0,i]=np.inf

    for i in range(1, N+1):
      start_j = max(1, i-r)
      end_j = min(M, i+r)+1

      for j in range(start_j, end_j):
        D_mat[i][j] = dist_mat[i-1][j-1] + min(D_mat[i-1][j], D_mat[i,j-1], D_mat[i-1][j-1])
  
    return  D_mat[N][M]

    ##n = len(ts1)
    ##cost_matrix = np.full((n, n), np.inf)
    ##cost_matrix[0, 0] = (ts1[0] - ts2[0]) ** 2
    
    ##if r is None:
    ##  r = max(n, n)
      
    ##for i in range(1, n):
    ##  for j in range(max(1, i - r), min(n, i + r)):
    ##    cost = (ts1[i] - ts2[j]) ** 2
    ##    cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])
 
    ##dtw_dist = cost_matrix[n - 1, n - 1]
    
    ##return dtw_dist