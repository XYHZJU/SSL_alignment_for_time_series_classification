from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
 
def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')    
    plt.title(title)    
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    
    plt.yticks(num_local, labels_name)    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./fig/'+title+'.png', format='png')

# def get_confusion_matrix()
# gt = []
# pre = []
# with open("result.txt", "r") as f:
#     for line in f:
#         line=line.rstrip()
#         words=line.split()
#         pre.append(int(words[0]))
#         gt.append(int(words[1]))
 
# cm=confusion_matrix(gt,pre)  
# print(cm)
# print('type=',type(cm))
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
# labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 
# plot_confusion_matrix(cm,labels,'confusion_matrix')  
 