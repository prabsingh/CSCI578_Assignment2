import scipy.io as sio
import scipy.stats as ss
import scipy.misc as smp
import numpy as np
import matplotlib.pyplot as plt
from math import *


#load the data
data = sio.loadmat('1bba1_ml.mat')

#create the training data
train_data = []
train_class = []

for i in range(0, 288000):
    col = data['xc'][i]
    Y = data['Y'][i][0]
    if (col < 420 and Y != 0):
        train_data.append(data['X'][i])
        train_class.append(Y)

dim = len(train_data[0]) 
n = len(train_data)
class_labels = list(set(train_class))
num_classes = len(class_labels)
class_labels.append(0)

#split the training data set into the different classes
split_train_data = [None] * num_classes
for i in range(0, num_classes):
    temp = []
    for j in range(0, n):
        if(train_class[j] == class_labels[i]):
            temp.append(train_data[j])
    split_train_data[i] = temp


#compute the common covariance
cov = np.zeros((num_classes, dim, dim))
for i in range(0, num_classes):
    cov[i] = np.cov(split_train_data[i], rowvar=False, bias=True)

cc = np.zeros((dim,dim))
for i in range(0, num_classes):
    cc += cov[i] 
    
cc /= num_classes


#compute the hyperparam mean as the mean of means
means = np.zeros((num_classes, dim))
for i in range(0, num_classes):
    means[i] = np.mean(split_train_data[i], axis=0)
           
#hyperparameters
mu = np.mean(means, axis = 0)
k_range = [.1, .5, 1, 5, 10, 50, 100]
m_range = [dim+2, 3*dim, 10*dim, 50*dim, 100*dim]
S_range = [.1, .5, 1, 5, 10, 50, 100]

k = 0
m = 0
s = 0


#compute cov_dif
cov_dif = np.zeros((num_classes, dim, dim))
for i in range(0, num_classes):
    cov_dif[i] = np.outer(mu - means[i], mu - means[i])

#calc params
params_mu = np.zeros((num_classes + 1, dim))
params_Sigma = np.zeros((num_classes + 1, dim, dim))
params_df = np.zeros(num_classes + 1)

def  calc_params(k, m, s):
    for i in range(0, num_classes):
        num_ele = len(split_train_data[i])
        x_bar = means[i]    
        params_mu[i] = loc_mean = ((k * mu) + (num_ele * x_bar))/(num_ele + k)

        df = (num_ele + m + 1 - dim)
        params_df[i] = df

        shape_scale = (num_ele + k + 1)/((num_ele + k) * df)
        shape_matrix = cc/s + ((num_ele -1) * cov[i]) + ((num_ele*k)/(num_ele + k) * cov_dif[i])

        params_Sigma[i] = shape_scale * shape_matrix
    params_mu[num_classes] = mu
    params_Sigma[num_classes] =  ((k + 1)/ (k *(m + 1 - dim))) * cc
    params_df[num_classes] = m + 1 - dim 

p0 = np.zeros(num_classes + 1)
def  simple():
    for i in range(0, num_classes + 1):
        d = dim
        df = params_df[i]
        Sigma = params_Sigma[i]
        top = lgamma(.5 * (d + df))
        p1 = lgamma(.5 * df)
        p2 = .5 * d * log(df * np.pi)
        sign, value = np.linalg.slogdet(Sigma)
        p3 = .5 * sign * value
        p0[i] = top - p1 - p2 - p3

inv = np.zeros((num_classes + 1, dim, dim))
def invert():
    for i in range(0, num_classes + 1):
        inv[i] = np.linalg.inv(params_Sigma[i])
    
    
def likelyhood(x, class_k):
    
    p4 = np.dot(x - params_mu[class_k],inv[class_k])
    p5 = np.dot(p4,(x-params_mu[class_k]))
    p6 = .5 * (params_df[class_k] + dim) * log(1 + (1.0/params_df[class_k]) * p5)
    return p0[class_k] - p6

def f1_score(confusion, size, label):
    tp = 0
    fn = 0
    fp = 0
    for i in range(0, size):
        if(label == i):
            tp += confusion[i][i]
        else:
            fn += confusion[label][i]
            fp += confusion[i][label]
    fn += confusion[label][size]
    if(tp == 0):
        return 0
    return (2 * tp)/((2* tp) + fn + fp)


'''def createImage():
    abc = np.zeros((450,640), dtype=np.uint8)
    c = 0
    for i in range(0,450):
        for j in range(0,640):
            abc[i][j] = data['Y'][c][0]
            c += 1
    plt.imshow(abc)
    plt.show()'''

#create the testing dataset
test_data = []
test_class = []
for i in range(0, 288000):
    col = data['xc'][i]
    Y = data['Y'][i][0]
    #if (col >= 420 and Y != 0):
    test_data.append(data['X'][i])
    test_class.append(Y)

def main():
    trial = 0
    f1_max = 0;
    k_max = None;
    s_max = None;
    m_max = None;
    oos_max = None;
    for a in k_range:
        for b in S_range:
            for c in m_range:
                print(trial)
                trial += 1
                k = a
                s = b
                m = c
                calc_params(k, m, s)
                simple()
                invert()
                confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
                #for each data point
                #find the maximum likelyhood
                #adjust the confusion_matrix accordingly
                for i in range(0, len(test_data)):
                    cur_max = -inf
                    cur_class = None
                    for j in range(0, num_classes + 1):
                        prob = likelyhood(test_data[i], j)
                        if(prob > cur_max):
                            cur_max = prob
                            cur_class = j
                    confusion_matrix[class_labels.index(test_class[i])][cur_class] += 1

                oos = 0
                for i in range(0, 18):
                    oos += confusion_matrix[i][17]

                f1_avg =  0
                for i in range(0, num_classes):
                    f1_avg += f1_score(confusion_matrix, 17, i)
                f1_avg /= num_classes

                if(f1_avg > f1_max):
                    f1_max = f1_avg
                    k_max = k
                    s_max = s
                    m_max = m
                    oos_max = oos

main()
