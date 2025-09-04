#!/usr/bin/env python
# coding: utf-8

# In[247]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


# In[248]:


data_file = 'promoters.data' 


# In[249]:


with open('promoters.data', 'r') as f:
    content = f.read()
    print(content[:500])


# In[250]:


sequences = []
labels = []

with open(data_file, 'r') as f:
    lines = f.read().strip().split('\n')

#since each record appears to be on a single line, parse line-by-line
for line in lines:
    line = line.strip()
    #split by comma to separate label and sequence info
    parts = line.split(',')

    #the first part contains the label (+ or -)
    label_char = parts[0].strip()
    label = 1 if label_char == '+' else 0

    #the sequence is in the last part (the ID is in the middle)
    sequence_part = parts[-1].strip()

    labels.append(label)
    sequences.append(sequence_part)


df = pd.DataFrame({'sequence': sequences, 'label': labels})
print(df.head())


# In[251]:


#how many sequences in the dataframe 
print(df.shape[0])


# In[252]:


def one_hot_encode_sequence(seq):
    mapping = {'a': [1,0,0,0],
               'c': [0,1,0,0],
               'g': [0,0,1,0],
               't': [0,0,0,1]}
    encoded = []
    for nucleotide in seq:
        encoded.extend(mapping[nucleotide])
    return encoded

X = np.array([one_hot_encode_sequence(s) for s in df['sequence']])
y = np.array(df['label'])


# In[253]:


#normalization 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[254]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[255]:


for k in [3,5,10]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, X_train_scaled, y_train_scaled, cv=kf, scoring='accuracy')
    print(f"{k}-fold CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


# In[256]:


#Logistic Regression 

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train_scaled)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]

print("Logistic Regression Accuracy:", accuracy_score(y_test_scaled, y_pred_lr))
print("Logistic Regression Precision:", precision_score(y_test_scaled, y_pred_lr))
print("Logistic Regression Recall:", recall_score(y_test_scaled, y_pred_lr))
print("Logistic Regression F1:", f1_score(y_test_scaled, y_pred_lr))
print("Logistic Regression ROC AUC:", roc_auc_score(y_test_scaled, y_prob_lr))


# In[257]:


#plot the confusion matrix 

cm_lr = confusion_matrix(y_test_scaled, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# In[258]:


#Naive Bayes 

nb = BernoulliNB()
nb.fit(X_train_scaled, y_train_scaled)

y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:,1]

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Precision:", precision_score(y_test, y_pred_nb))
print("Naive Bayes Recall:", recall_score(y_test, y_pred_nb))
print("Naive Bayes F1:", f1_score(y_test, y_pred_nb))
print("Naive Bayes ROC AUC:", roc_auc_score(y_test, y_prob_nb))


# In[259]:


cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Confusion Matrix")
plt.show()


# In[260]:


#upload new ecoli strain genome ( E. coli K-12 MG1655)

genome_file = 'GCF_000005845.2_ASM584v2_genomic.fna' 


# In[261]:


with open('GCF_000005845.2_ASM584v2_genomic.fna', 'r') as f:
    content = f.read()
    print(content[:500])  # Print the first 500 characters


# In[262]:


#skip the first line that starts with '>', and join all remaining lines into one string
with open(genome_file, 'r') as f:
    lines = f.read().strip().split('\n')
    description = lines[0]
    sequence = ''.join(lines[1:]) 


# In[263]:


print(len(sequence))


# In[264]:


#break into 57-bp groups to match the original dataset 

segment_length = 57
segments = []
for start in range(0, len(sequence) - segment_length + 1, segment_length):
    segment = sequence[start:start+segment_length]
    segments.append(segment)

print("Number of segments extracted:", len(segments))


# In[265]:


def one_hot_encode_sequence(seq):
    mapping = {'A': [1,0,0,0],
               'C': [0,1,0,0],
               'G': [0,0,1,0],
               'T': [0,0,0,1]}
    encoded = []
    for nucleotide in seq:
        encoded.extend(mapping[nucleotide])
    return encoded

X_new = np.array([one_hot_encode_sequence(s) for s in segments])


# In[266]:


predictions_nb = nb.predict(X_new)
#print(predictions_nb[:500])


# In[267]:


promoter_count = (predictions_nb == 1).sum()
non_promoter_count = (predictions_nb == 0).sum()

plt.bar(['Non-Promoter', 'Promoter'], [non_promoter_count, promoter_count], color=['blue', 'green'])
plt.title('Naive Bayes Predicted Class Distribution')
plt.ylabel('Count')
plt.show()


# In[268]:


predictions_lr = lr.predict(X_new)
#print(predictions_lr[:500])


# In[269]:


promoter_count = (predictions_lr == 1).sum()
non_promoter_count = (predictions_lr == 0).sum()

plt.bar(['Non-Promoter', 'Promoter'], [non_promoter_count, promoter_count], color=['blue', 'green'])
plt.title('Logistic Regression Predicted Class Distribution')
plt.ylabel('Count')
plt.show()


# In[270]:



unique, counts = np.unique(predictions_lr, return_counts=True)
print(dict(zip(unique, counts)))


# In[271]:



unique, counts = np.unique(predictions_nb, return_counts=True)
print(dict(zip(unique, counts)))


# In[ ]:




