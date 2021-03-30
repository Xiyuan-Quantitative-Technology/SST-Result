# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:00:03 2020

@author: 86178
"""

import numpy as np

def pad_adjacency_matrix(adj_list, pad_value):
    #generating adjacency matrix list
    mat_dim = [len(adj) for adj in adj_list]
    
    adj_matrix_list = []
    for i in range(len(mat_dim)):
        adj_matrix = np.zeros((mat_dim[i], mat_dim[i]))
        
        for m_id in range(mat_dim[i]):
            adj_matrix[m_id, adj_list[i][m_id]] = 1
            
        adj_matrix_list.append(adj_matrix)
        
    
    """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
    pad_mat_dim = sum(mat_dim)
    zeros = np.zeros((pad_mat_dim, pad_mat_dim))
    pad_adj_mat = pad_value + zeros
    
    j = 0
    for k, matrix in enumerate(adj_matrix_list):
        n = mat_dim[k]
        pad_adj_mat[j:j+n, j:j+n] = matrix
        j += n

    
    #Symmetric normalization of padded adj matrix
    #making adjacency hat matrix
    I = np.eye(pad_mat_dim)
    pad_adj_mat_hat = pad_adj_mat + I
    
    #degree matrix
    pad_degree_list = np.sum(pad_adj_mat_hat, 1)
    pad_degree_mat = np.diag(pad_degree_list)
    
    # Inverse of the Cholesky decomposition of degree matrix
    Inv_Chol_pad_Dmat = np.linalg.cholesky(pad_degree_mat)
    Inv_Chol_pad_Dmat = np.linalg.inv(Inv_Chol_pad_Dmat)
    
    #Symmetric normalization of adj matrix
    pad_adj_mat_norm = Inv_Chol_pad_Dmat.dot(pad_adj_mat_hat).dot(Inv_Chol_pad_Dmat)
    
    return pad_adj_mat_norm, mat_dim



