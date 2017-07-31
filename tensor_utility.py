import ncp
import json
import numpy as np
from numpy import zeros, ones, diff, kron, tile, any, all, linalg
import numpy.linalg as nla
import time
from sktensor import ktensor,dtensor
from numpy.random import rand
def get_index_label_dict(labels):
    index_label_dict={}
    for ind,la in enumerate(labels):
        index_label_dict[la]=ind
    return index_label_dict,list(labels)

def load_pickle(f_name):
    with open(f_name, 'rb') as handle:
        b = json.load(handle)
    return b[0],b[1],b[2]
def get_doc_components(doc_labels):
    doc_num=len(set(doc_labels))
    doc_components=[]
    for i in range(doc_num):doc_components.append([])
    for i in range(len(doc_labels)):doc_components[doc_labels[i]].append(i)
    return doc_components

def get_doc_tensor(matrix,labels,doc_labels):
    doc_components=get_doc_components(doc_labels)
    label_num=len(list(set(labels)))
    tensor=np.zeros(shape=(len(doc_components),label_num,label_num))
    for d in range(len(doc_components)):
        for i in doc_components[d]:
            for j in range(len(matrix[i])):
                if matrix[i][j]==0:continue
                tensor[d][labels[i]][labels[j]]+=1
    return tensor
def uk_product(X,Y):

    F=np.zeros(shape=(len(X)*len(Y),len(X[0])))
    for i in range(len(X)):
        for j in range(len(X)):
            for k in range(len(X[0])):
                F[i*len(X)+j][k]=X[i][k]*Y[j][k]
    return F
def get_component_pair_list(A,B,r,threshold=0.95):
    component_list=[]
    for i in range(r):
        matrix=np.outer(A[:,i],B[:,i])
        ##eliminate not important members
       # s=float(sum(matrix[j]))
        s=sum(map(sum, matrix))
        for j in range(len(matrix)):
            if s==0:continue
            for k in range(len(matrix[0])):
                if float(matrix[j][k])/float(s)<(1-threshold):
                    matrix[j][k]=0.0
        ##
        component_list.append(matrix)
    return component_list
def print_component_labels(component_id,X_approx_ks,lset):
    print '######component id:'+str(component_id)
    print '#in_labels#'
    for i in range(len(X_approx_ks.U[1])):
        if X_approx_ks.U[1][i][component_id]>0:
            print lset[i],X_approx_ks.U[1][i][component_id]
    print 
    print '#out_labels#'
    for i in range(len(X_approx_ks.U[2])):
        if X_approx_ks.U[2][i][component_id]>0:
            print lset[i],X_approx_ks.U[1][i][component_id]
def print_component_pairs(component_id,component_pair_list,lset,label_num):
    print '######component id:'+str(component_id)
    for i in range(label_num):
        for j in range(label_num):
            if component_pair_list[component_id][i][j]>0:
                print lset[i],'->',lset[j],component_pair_list[component_id][i][j]

