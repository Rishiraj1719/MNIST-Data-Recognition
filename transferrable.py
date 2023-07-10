import numpy as np
import pandas as pd
from matplotlib import pyplot as pt
import math as ma
## All Checked 
### break
# list_ex_1=[np.array([i for i in range(5)]),np.array([i**2 for i in range(3)])]
# print(list_ex_1)
# list_ex_1[2]=np.array([2,7,8,4])
# print(list_ex_1)
### break

# Converting the dataset in the required form
global row_pixel, col_pixel

df_base=pd.read_csv('train.csv')
df=np.array(df_base.iloc[:,1:])
target=np.array(df_base.iloc[:,0])
df_sh=df.shape
row_pixel=col_pixel=int(ma.sqrt(df_sh[1]))
df=df.reshape(df_sh[0],row_pixel,col_pixel)
df_sh=df.shape
m=df.shape[0]
#  m is int not ndarray int type
## All Checked
### break

output=np.array([i for i in range(10)])
# to print an image
def show_img(array):
    pt.imshow(array,cmap='gray')
    pt.show()
def pad_img(array,p):
    ar=array.copy()
    n=row_pixel
    h_pad_0=np.zeros((n,1), dtype=int)
    y=np.hstack([h_pad_0]*p)
    x=np.hstack((y,array,y))
    v_pad_1=np.zeros((1,n+2*p), dtype=int)
    y=np.vstack([v_pad_1]*p)
    x=np.vstack((y,x,y))
    return x
# a_1=pad_img(df[1],2)
# print(a_1.shape)
### break
# Example of nn_structure variable
nn_struc_c=[[3,(3,3),1,(2,2),2],[1,(3,3),1,(2,2),1]]
# ---------_^
# Layers marked
# format of variables (below)
# [no. of layers->[filters,(filter_row,filter_row),padding,(pol_filter_row,pol_filter_row),pol_stride]...]
print(nn_struc_c)

"""
A single layer's information as we want to get the idea of convolution layers
For feature layers we define a separate structure  
"""

nn_struc_f=[100,40,10]
### break
def bytesize(lis):
    l=len(lis)
    c=0
    for i in range(l):
        c+=lis[i].size*lis[i].itemsize
    return c
### break
# To generate the list of array of parameters for all convolution layers
def c_para_constr(rows,nn_struc_c):
    row_iter=rows
    c_layers=len(nn_struc_c)
    c_wt_list=[]
    c_b_list=[]
    c_pol_list=[]
    for i in range(c_layers):
        filters=nn_struc_c[i][0]
        conv_filter_row=nn_struc_c[i][1][0]
        padding=nn_struc_c[i][2]
        pol_stride=nn_struc_c[i][4]
        new_row=ma.floor((row_iter-conv_filter_row+2*padding)/pol_stride)+1
        c_wt_list.append(np.random.rand(filters,new_row,new_row))
        c_b_list.append(np.random.rand(filters,new_row,new_row))
        pol_element=np.array([nn_struc_c[i][3],nn_struc_c[i][4]])
        c_pol_list.append(pol_element)
        row_iter=new_row
    total_features=c_b_list[-1].size
    return c_wt_list,c_b_list,c_pol_list,total_features

# To generate the list of array of parameters for all feature layers 
def f_para_constr(nn_struc_f,init_features):
    f_wt_list=[]
    f_b_list=[]
    features=init_features
    f_layers=len(nn_struc_f)
    for i in range(f_layers):
        f_wt_list.append(np.random.rand(nn_struc_f,features))
        f_b_list.append(np.random.rand(features,1))
        features=nn_struc_f[i]
    return f_wt_list,f_b_list
### break
def nn_make(nn_struc_c,nn_struc_f,c_b,f_b):
    list_c=[]
    ar_f=np.array(nn_struc_f)
    list_all=[]
    layer_c=len(c_b)
    layer_f=len(f_b)
    n=row_pixel
    for i in range (layer_c):
        element=[]
        ax_0=nn_struc_c[i][0]
        c_row=nn_struc_c[i][1][0]
        padding=nn_struc_c[i][2]
        pol_row=nn_struc_c[i][3][0]
        pol_stride=nn_struc_c[i][4]
        ax_1=ma.floor((n-c_row+2*padding)/pol_stride)+1
        new_dim_c=(ax_0,ax_1,ax_1)
        ax_1=ma.floor((ax_1-pol_row)/pol_stride)+1
        new_dim_p=(ax_0,ax_1,ax_1)
        element.append(new_dim_c)
        element.append(new_dim_p)
        list_c.append(element)
        n=ax_1
    init_features=element[1].size
    list_all.append(list_c)
    list_all.append(init_features)
    list_all.append(ar_f)
    return list_all
### break
a_001,a_002,a_003=c_para_constr(row_pixel,nn_struc_c)
print(a_001,a_002,a_003)
print(bytesize(a_001)+bytesize(a_002))
### break
# Rectified Linear Unit
def ReLu(array):
    array_0=np.zeros(array.shape,dtype=int)    
    array=np.maximum(array,array_0)
    return array

def softmax(array):
    array=np.exp(array)
    array=array/(np.sum(array))
    return array
# For the activation

def activation(array,func):
    if func=='ReLu':
        array=ReLu(array)
    elif func=='sigmoid':
        array=1/(1+np.exp((-1)*array))
    elif func=='tanh':
        array=np.tanh(array)
    elif func=='softmax':
        array=softmax(array)
    return array
### break
# Convolution Layer
def conv(dat,c_wt_matrix,c_b_matrix,c_pol_matrix):
    fs=
    r=fs[0]
    n=row_pixel
    num_filter=nn_struc_c[0]
    ar=np.zeros((num_filter,n-r+1,n-r+1),dtype=int)
    for k in range(num_filter):
        for i in range(r):
            for j in range(r):
                ar+=filter[i,j]*dat[i:(n-r+i+1),j:(n-r+j-1)]
    return ar

# Pooling layer
def max_pol(dat,filter,stride):
    pol=np.zeros(np.shape(dat[:,::stride,::stride]),dtype=int)
    for k in range(dat.shape[0]):
        for i in range (row_pixel):
            for j in range (col_pixel):
                pol=np.maximum(dat[k,i::stride,j::stride],pol[k,:,:])
    return pol

# Transforming laer consisting both conv() and max_pol() 
def conv_tr_layer(dat,c_wt_list,pol_filter,stride_pol):
    ar=dat.copy()
    ar=conv(ar,conv_filter)
    ar=max_pol(ar,pol_filter,stride_pol)
    return ar

# For forward neural network propagation over a single layer
def nnl_fwd_layer(prev,num_new):
    theta_matrix=np.random.random(num_new,prev)
    bias=np.random(num_new,1)
    new=np.add((theta_matrix@prev),bias)
    new=activation(new)
    return new
### break

# For forward propagation over decided layers
def forward_prop(dat,c_wt_list,c_b_list,pol_f_list,f_wt_list,f_b_list):
    l_c=len(nn_struc_c)
    ar=dat.copy()
    for i in range(l_c):
        new_ar=conv_tr_layer(ar,c_wt_list[i],c_b_list[i],)
        ar=new_ar
        
    return

# For backward neural network propagation
def back_prop():

    return

