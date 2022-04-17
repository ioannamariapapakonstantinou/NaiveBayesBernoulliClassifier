

#NaiveBayesBernoulliClassifier imdb_dataset
#ArtificialIntelligence 5th Semester
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
#from google.colab import files #gia to download twn diagrammatwn








def main(number, size):
    number = number + 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=number, skip_top=10)
   
    x_train = x_train[:size]
    y_train = y_train[:size]
    
   
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    k = list(word_index.values())
   
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'

    x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
    x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

    "Creating the vocabulary"
    vocabulary = list()
    for text in x_train:
        tokens = text.split()
        vocabulary.extend(tokens)
    vocabulary = set(vocabulary)

    # metatroph se list gia na exoun sygkekrimenh diataksh oi lekseis
    vocabulary = list(vocabulary)
    vocabulary.remove('[oov]')
    
    
    return vocabulary, x_train, y_train, x_test, y_test, k

def h_function(vocabulary, x_train, y_train):
    x_train_binary = list()
    
###############################################################################
    for text in tqdm(x_train):
        tokens = text.split()
        binary_vector = list()
        for vocab_token in vocabulary:
            if vocab_token in tokens:
                binary_vector.append(1)
            else:
                binary_vector.append(0)

        x_train_binary.append(binary_vector)

    x_train_binary = np.array(x_train_binary)
    x_train_b = x_train_binary
    
   
    dataset = dict()
    temp = list()
    for j in range(0, x_train_binary.shape[1]):
        for i in range(0, x_train_binary.shape[0]):
            temp.append(x_train_binary[i][j])
        dataset[vocabulary[j]] = copy.deepcopy(temp)

        temp.clear()

    for i in range(0, len(y_train)):
        temp.append(y_train[i])
    dataset['Review_Sol'] = copy.deepcopy(temp)
    
    return x_train, y_train, x_train_b, vocabulary, dataset
########################################################################
def h_function_t(vocabulary, x_test, y_test):  
    x_test_binary = list()
    for text in tqdm(x_test):
        tokens = text.split()
        binary_vector2 = list()
        for vocab_token in vocabulary:
            if vocab_token in tokens:
                binary_vector2.append(1)
            else:
                binary_vector2.append(0)
        x_test_binary.append(binary_vector2)

    x_test_binary = np.array(x_test_binary)
    x_test_b = x_test_binary
    
    dataset2 = dict()
    temp2 = list()
    for j in range(0, x_test_binary.shape[1]):
        for i in range(0, x_test_binary.shape[0]):
            temp2.append(x_test_binary[i][j])
        dataset2[vocabulary[j]] = copy.deepcopy(temp2)

        temp2.clear()

    for i in range(0, len(y_test)):
        temp2.append(y_test[i])
    dataset2['Review_Sol'] = copy.deepcopy(temp2)
    
    return x_test, y_test, x_test_b, dataset2
###############################################################################
def print_results(dataset, dataset2, number1, x_train): 
       
    train_data = pd.DataFrame.from_dict(dataset)
    test_data = pd.DataFrame.from_dict(dataset2)
    
    print(train_data)
    print(test_data)
    
    return 0


#...........................................................

class Naive_Bayes:
    
  possYN = [] #possibility
  sumYN = [] #sum
  possPos = [] #P(word1|positive)
  possNeg = [] #P(word1|negative)
  results = []
  
  
  def __init__(self, x_train, y_train, x_train_b, x_test_b, y_test,x_test, voc):
    
    self.x_train_b = x_train_b
    self.x_train = x_train
    self.y_train = y_train
    self.x_test_b = x_test_b
    self.y_test = y_test
    self.x_test = x_test
    self.voc = voc
    
    
  
  #possibility P(positive), P(negative)
  def calc_pX(self):
    self.sumYN.clear()
    self.possYN.clear()
    count = 0
    
    for a in self.y_train:
      if (0 == a):
        count+=1
   
    self.possYN.append(count/self.x_train.size) #P(Negative)
    self.possYN.append(self.x_train.size-count/self.x_train.size) #P(positive)
    self.sumYN.append(count) #sum of Negative reviews
    self.sumYN.append(self.x_train.size - count) #sum of Positive reviews
    
    return 0

  
  # P(word1|positive) = sumOf(word1->exists AND review->positive)/sumOfPositiveReviews
  def calc_pXA(self, x_tr_b,y):
    self.possPos.clear()
    self.possNeg.clear()
    s = int(x_tr_b[0].size)
    
    for a in range(s):# gia oles tis lekseis
      count = 0
     
      for b in range(int(y.size)): #gia ola ta vector
        
        if (x_tr_b[b][a] == 1 and y[b].astype(int) == 1):
          count+=1
          continue
      
      w = (count+1)/(self.sumYN[1]+2) # P(word1|positive) edw epishs Laplace
      self.possPos.append(w)
      
  ###########################################################################    
   
    for a in range(s):# gia oles tis lekseis
      count = 0
     
      for b in range(int(len(x_tr_b))): #gia ola ta vector
        if (x_tr_b[b][a] == 1 and y[b].astype(int) == 0):
          count+=1
          continue
      w = (count+1)/(self.sumYN[0]+2) # P(word1|negative)
      self.possNeg.append(w)
    return 0
    
  

  # P(positive|phrase) = log10(P(phrase|positive)) + log10(P(positive))
  
  def calc_phrase(self, x_t_b, y):
   
    self.results.clear()
    g = 0
    h = 0
    totalPossibilityN = 0 #P(phrase|negative)
    totalPossibilityP = 0 #P(phrase|positive)
   
    for a in range(int(len(x_t_b[0]))): #gia olew tis lekseis
        
      k = x_t_b[y][a]
  
      prop = ((self.possNeg[a])**k)*((1-self.possNeg[a])**(1-k)) 
      
      if (prop>0):
        totalPossibilityN += np.log10(prop, where =  prop>0)
      prop1 = ((self.possPos[a])**k)*((1-self.possPos[a])**(1-k))
      if (prop1>0):
        totalPossibilityP += np.log10(prop1, where = prop1>0)
      
        
    g = totalPossibilityP + math.log(self.possYN[1])  #positive
    h = totalPossibilityN + math.log(self.possYN[0])  #negative
    
     
    if (g>h):
      #"Positive"
      return 1
    else:
      #"Negative"
      return 0
    
    
  def fit(self, X, y):
    self.calc_pX()
    self.calc_pXA(X, y)
    return self
 
  def predict(self, v):
    res = [self.calc_phrase(v, b) for b in range(int(len(v)))] #gia ola ta vector/paradeigmata
    return np.array(res)
    
  def ac_pr_re_f1(self, results, y):
    count = 0 #count
    fp = 0 # false positive
    fn = 0 # false negative
    tp = 0 # true positive
    for i in range(len(results)):
      if results[i] == y[i]: # accuracy
        count+=1
      if results[i]==1 and y[i]==0:#precision
        fp+=1
      if results[i]==0 and y[i]==1:#recall 
        fn+=1
      if results[i] == y[i] == 1:
        tp+=1
    
    acc = count/len(results)
    pr  = tp / (tp + fp)
    re  = tp / (tp + fn)
    
    if (pr+re>0):
      f1 = (2*re*pr)/(pr+re)
    else:
      f1 = 0
    return acc, pr, re, f1
 
  
    

#...........................................................
def calculateIG(x_train_b, voc, y_train):

  numOfExamples = len(x_train_b) # get the num of vectors
    
  numOfFeatures = len(voc) # get the num of columns

  IG = [None] *numOfFeatures# IG for every feature
  positives = 0
  for i in y_train:
      
      
     if (i == 1):
       positives+=1 # count how many are C=1
      

    # We need all of the below for the IG formula


  PC1 = positives / numOfExamples# P(positive)
   
  HC = twoCEntropy(PC1)# H(C)

  #P(X=1) --> prob. of X=1
  #P(X=0) --> 1 - P(X=1)
  PX1 = [None]*numOfFeatures

  #P(C=1|X=1)
  #P(C=0|X=1) = 1 - P(C=1|X=1)
  PC1X1 = [None]*numOfFeatures

  #P(C=1|X=0)
  #P(C=0|X=0) = 1 - P(C=1|X=0)
  PC1X0 = [None]*numOfFeatures

  #H(C=1|X=1)
  HCX1 = [None]*numOfFeatures

  #H(C=1|X=0)
  HCX0 = [None]*numOfFeatures
  

  for  j in range(numOfFeatures):
      
    cX1 = 0 # for every feature, count the examples in which X=1
    cC1X1 = 0 # count how many examples are C=1 given X=1
    cC1X0 = 0# count how many examples are C=1 given X=0
    for i in range(numOfExamples):
      if (x_train_b[i][j] == 1):
        cX1+=1
      if(x_train_b[i][j] == 1 and y_train[i].astype(int) == 1):
        cC1X1+=1
      if(x_train_b[i][j] == 0 and y_train[i].astype(int) == 1): 
        cC1X0+=1		

      PX1[j] = cX1 / numOfExamples# P(X=1) for j-th feature.

      #if all examples have X=0
      if(cX1 == 0): 
        PC1X1[j] = 0.0# no example has X=1 so P(C=1|X=1) = 0
      else:
        PC1X1[j] = cC1X1 / cX1 # dividing by how many examples have X=1 in general
      # if all examples have X=1
      if(cX1 == numOfExamples):
        PC1X0[j] = 0.0 # no example has X=0 so P(C=1|X=0) = 0
      else: 
        PC1X0[j] = cC1X0 / (numOfExamples - cX1) # dividing by how many examples have X=0 in general

      HCX1[j] = twoCEntropy(PC1X1[j])# entropy for the category when X=1 (for j-th feature)
      HCX0[j] = twoCEntropy(PC1X0[j])# entropy for the category when X=0 (for j-th feature)

      IG[j] = HC - ( (PX1[j] * HCX1[j]) + ( (1.0 - PX1[j]) * HCX0[j]) )# IG formula 

      

  return IG
 # calculate entropy for two categories.
def twoCEntropy(cProb):
  if (cProb == 0 or cProb == 1):
    return 0.0
  else:
    return - ( cProb * math.log2(cProb) ) - ( (1.0 - cProb) * math.log2(1.0 - cProb))	
 
    

if __name__ == '__main__':
  
  number = int(input("Enter the desired number of columns: "))
  x_variable = [] #training set size
  y_variable = [] #%accuracy train
  z_variable = [] #%accuracy test

  pr1_variable_tr = [] #precision train
  pr1_variable_t  = [] #precision test

  re1_variable_tr = [] #recall train
  re1_variable_t  = [] #recall test

  f1_variable_tr = [] #f1 train
  f1_variable_t  = [] #f1 test


  for i in range(5000, 27500, 2500):
    x_variable.append(i)
    voc, x, y, x_t, y_t, words = main(number, i)
    x_tr, y_tr, x_tr_b, voc, data = h_function(voc, x, y)
    
    d = calculateIG(x_tr_b, voc, y_tr)
  

    k = [voc for _, voc in sorted(zip(d, voc))] #sort to leksiko analoga me to IG se afksousa seira 
    
    
    k.reverse() # antistrefo to leksiko gia na exw to leksiko me f8inon IG, more details sto report
    
    k = k[0:int(len(k)/2+7)]
  
    x_t1, y_t1, x1, voc, data = h_function(k, x, y)
    
    x_test, y_test, x_test_binary, data2 = h_function_t(k, x_t, y_t)

    a1 = Naive_Bayes(x_tr, y_tr, x1, x_test_binary, y_test,x_test, k)
    
    a1.fit(x1, y_tr)
    #train data
    td = a1.predict(x1)#gia tis kampyles alliws den xreiazetai
    ac1, pr1, re1, f1 = a1.ac_pr_re_f1(td, y_tr)
    y_variable.append(ac1)
    pr1_variable_tr.append(pr1)
    re1_variable_tr.append(re1)
    f1_variable_tr.append(f1)
    
    #test data
    test_preds = a1.predict(x_test_binary)
    ac2, pr2, re2, f2 = a1.ac_pr_re_f1(test_preds, y_t)
    z_variable.append(ac2)
    pr1_variable_t.append(pr2)
    re1_variable_t.append(re2)
    f1_variable_t.append(f2)
    print_results(y,y_t,1,x_tr)
  #accuracy train data
 
  pa = plt.figure()
  plt.plot(x_variable, y_variable, label = 'Training Data')
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Accuracy")
  plt.show()
  pa.savefig('AccTR.png')
  #files.download('AccTR.png')
  dataset = pd.DataFrame({'Training Set Size':x_variable , 'Accuracy training data': y_variable}, columns=['Training Set Size', 'Accuracy training data'])
  print(dataset)
  #accuracy test data
  
  pa1 = plt.figure()
  plt.plot(x_variable, z_variable, label = 'Testing Data' )
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Accuracy")
  plt.show()
  pa1.savefig('AccTE.png')
  #files.download('AccTE.png')
  dataset1 = pd.DataFrame({'Training Set Size':x_variable , 'Accuracy testing data': z_variable}, columns=['Training Set Size', 'Accuracy testing data'])
  print(dataset1)
  
  #precision train data
  pa2 = plt.figure() 
  plt.plot(x_variable, pr1_variable_tr, label = 'Training Data')
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Precision")
  plt.show()
  pa2.savefig('PrTR.png')
  #files.download('PrTR.png')
  dataset2 = pd.DataFrame({'Training Set Size':x_variable , 'Precision training data': pr1_variable_tr}, columns=['Training Set Size', 'Precision training data'])
  print(dataset2)

  #precision test data
  pa3 = plt.figure()
  plt.plot(x_variable, pr1_variable_t, label = 'Testing Data' )
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Precision")
  plt.show()
  pa3.savefig('PrTE.png')
  #files.download('PrTE.png')
  dataset3 = pd.DataFrame({'Training Set Size':x_variable , 'Precision testing data': pr1_variable_t}, columns=['Training Set Size', 'Precision testing data'])
  print(dataset3)

  #
  #recall train data
  pa4 = plt.figure()
  plt.plot(x_variable, re1_variable_tr, label = 'Training Data')
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Recall")
  plt.show()
  pa4.savefig('ReTR.png')
  #files.download('ReTR.png')
  dataset4 = pd.DataFrame({'Training Set Size':x_variable , 'Recall training data': re1_variable_tr}, columns=['Training Set Size', 'Recall training data'])
  print(dataset4)

  #recall test data
  pa5 = plt.figure()
  plt.plot(x_variable, re1_variable_t, label = 'Testing Data' )
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("Recall")
  plt.show()
  pa5.savefig('ReTE.png')
  #files.download('ReTE.png')
  dataset5 = pd.DataFrame({'Training Set Size':x_variable , 'Recall testing data': re1_variable_t}, columns=['Training Set Size', 'Recall testing data'])
  print(dataset5)

  #F1 train data
  pa6 = plt.figure()
  plt.plot(x_variable, f1_variable_tr, label = 'Training Data')
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("F1")
  plt.show()
  pa6.savefig('FTR.png')
  #files.download('FTR.png')
  dataset6 = pd.DataFrame({'Training Set Size':x_variable , 'F1 training data': f1_variable_tr}, columns=['Training Set Size', 'F1 training data'])
  print(dataset6)
  #F1 test data
  pa7 = plt.figure()
  plt.plot(x_variable, f1_variable_t, label = 'Testing Data' )
  plt.legend()
  plt.xlabel("Training Set Size")
  plt.ylabel("F1")
  plt.show()
  pa7.savefig('FTE.png')
  #files.download('FTE.png')
  dataset7 = pd.DataFrame({'Training Set Size':x_variable , 'F1 testing data': f1_variable_t}, columns=['Training Set Size', 'F1 testing data'])
  print(dataset7)

  print("THE END")