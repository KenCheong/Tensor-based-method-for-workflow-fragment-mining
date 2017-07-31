import ncp
from tensor_utility import *
from sktensor import ktensor,dtensor

####load tensor
matrix,labels,doc_labels=load_pickle('WC2.pickle')
index_label_dict,lset=get_index_label_dict(set(labels))
int_labels=[index_label_dict[la] for la in labels]
label_num=len(list(set(labels)))
tensor=get_doc_tensor(matrix,int_labels,doc_labels)
####
print 'tensor shape(workflow_num,labels,labels):',tensor.shape
r=25##number of components
X_approx_ks = ncp.nonnegative_tensor_factorization(dtensor(tensor), r)
component_pair_list=get_component_pair_list(X_approx_ks.U[1],X_approx_ks.U[2],r,threshold=0.95)


#### print first five components
for i in range(5):
    print_component_pairs(i,component_pair_list,lset,label_num)
####

'''
print 
for i in range(2):
    print_component_labels(i,X_approx_ks,lset)
'''

