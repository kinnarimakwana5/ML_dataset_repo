#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data analysis and manupilation tool
import os # method for interacting with operating system
import matplotlib.pyplot as plt #Python 2D plotting library which produces publication-quality figures + creates a plotting area in a figure 
import matplotlib.axis as ax #figure containing a single Axes
import numpy as np # working with arrays
import seaborn as sns # visualization
from scipy import stats #solving many mathematical equations and algorithms
from sklearn.decomposition import PCA #machine learning in Python
from sklearn.preprocessing import StandardScaler #emoving the mean and scaling to unit variance
from sklearn.metrics import confusion_matrix, accuracy_score, confusion_matrix, classification_report




# In[2]:


#Pandas converts 'NA' and null to NaN
#Adding missing values in a list that pandas is unable to identify
missing_values_format = ["n.a", "?", "NA", "n/a", "an", "--"]

#import dataset
df = pd.read_csv("data.txt")

#Checking the dimensions of the data
print("Breast cancer data set dimensions:", df.shape)


# In[3]:


df.head(10)


# In[4]:


df.tail(5)


# In[5]:


#check missing value in column
df.isnull().sum()


# In[6]:


len(df)


# In[7]:


#fraction of missing values in each column 
df.isnull().sum()*100/len(df)


# In[8]:


#Getting the summary statistics 
df["radius_mean"].describe()


# In[9]:


#Histogram and boxplot to visualize the distribution of the data and detection of outliers

sns.set(style="ticks")
f, (ax_box, ax_hist)= plt.subplots(2,sharex=True, gridspec_kw={"height_ratios":(.20,.90)})
sns.boxplot(df["radius_mean"], ax= ax_box, color="blue")
sns.histplot(df["radius_mean"], ax =ax_hist, kde=True, color="green")
ax_box.set(ysticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box)


# In[23]:


df["radius_mean"].fillna(df['radius_mean'].median(), inplace=True)

#Confirming whether the missing values are filled with the median
df["radius_mean"].isnull().sum()

#Checking the effect of replacement of missing values with median on the data 
df["radius_mean"].describe()


                         


# In[33]:


#Checking to see the effect of removal of extreme outliers
df_outlinerRemove = df.loc[df['area_mean'] < 1500]
df_outlinerRemove["area_mean"].describe()


# In[35]:


df['area_mean'].fillna(df['area_mean'].median(), inplace=True)
df.isnull().sum()


# In[19]:


#Storing a copy if needed at a later stage (before mapping the values for the diagnosis column)
cell_df=df.copy()


# In[20]:


#Distribution plot of the features
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])


# In[21]:


#Histogram plots of all the features
plt.figure(figsize = (20,20))
plotnumber = 1

print('\033[1mFeatures Distribution'.center(120))
for column in df.iloc[:,2:]:
    if plotnumber <=30:
        ax =plt.subplot(5,6,plotnumber)
        sns.histplot(df[column], kde =True, stat = 'density', color ="blue")
        plt.xlabel(column)

    plotnumber +=1
plt.tight_layout()
plt.show()


# In[22]:


#Checking for the count of unique values under "diagnosis"
df["diagnosis"].value_counts()


# In[23]:


#Summary statistics of the features
df.describe()


# In[24]:


#Normalisation using min-max scaling
from sklearn.preprocessing import MinMaxScaler 
norm = MinMaxScaler().fit(df.iloc[:,1:])
norm_df = norm.transform(df.iloc[:,1:])
norm_df = pd.DataFrame(norm_df, columns = df.iloc[:,1:].columns)
norm_df.head()


# In[46]:


#Heatmap

print('\033[1mHeatmap with features in rows and observations in columns'.center(100))
fig, ax = plt.subplots(figsize=(10,10))   
sns.heatmap(norm_df.T, cmap = "RdPu" ,  mask=None,)
norm_df.T.shape


# In[25]:


#Reordering the columns such that the samples from the same class are grouped together
    
fig, ax = plt.subplots(figsize=(10,10))
NormSortDf = norm_df.sort_values(by = 'diagnosis')
sns.heatmap(NormSortDf.T, cmap = "RdPu")


# In[27]:


#k-means clustering on the rows and rearranging them so that features from the same group are together

from sklearn.cluster import KMeans
kmeans = KMeans(2)
kmeans.fit(norm_df.iloc[:,1:]) #Includes only features; Excludes "diagnosis" 
identified_clusters = kmeans.fit_predict(norm_df.iloc[:,1:])
len(identified_clusters)
norm_df["KmeansCluster"] = identified_clusters
norm_dfKMSort = norm_df.sort_values("KmeansCluster")


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(norm_dfKMSort.T, cmap= "RdPu")


# In[28]:


#Using normalised data

data = norm_df.iloc[:,:11]
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(10,10))
g = sns.boxplot(x="features", y="value", hue="diagnosis", data=data , color = "skyblue")
label_0 = 'Malignant'
g.legend_.texts[0].set_text(label_0)
label_1 = "Benign"
g.legend_.texts[1].set_text(label_1)
plt.xticks(rotation=90)


# In[29]:


data = pd.concat([norm_df["diagnosis"],norm_df.iloc[:,11:21]], axis = 1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(10,10))
g = sns.boxplot(x="features", y="value", hue="diagnosis", data=data , color = "skyblue")
label_0 = 'Benign'
g.legend_.texts[0].set_text(label_0)
label_1 = "Malignant"
g.legend_.texts[1].set_text(label_1)
plt.xticks(rotation=90)


# In[35]:


data = pd.concat([norm_df["diagnosis"],norm_df.iloc[:,21:31]], axis = 1)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
plt.figure(figsize=(10,10))
g = sns.boxplot(x="features", y="value", hue="diagnosis", data=data , color = "skyblue")
label_0 = 'Benign'
g.legend_.texts[0].set_text(label_0)
label_1 = "Malignant"
g.legend_.texts[1].set_text(label_1)
plt.xticks(rotation=90)


# In[32]:


#T-test

t_table = pd.DataFrame()
columns = cell_df.iloc[:,2:].columns
for col in columns:
    new = pd.DataFrame(data=cell_df[[col, "diagnosis"]])
    new = new.set_index("diagnosis")
    t = pd.Series(stats.ttest_ind(new.loc['M'], new.loc['B']))
    t_table = t_table.append(t, ignore_index = True)
t_table.insert(0, "feature", cell_df.iloc[:,2:].columns, True)
t_table.columns = ["features", "t-statistics","p-value"]
t_table.head()


# In[33]:


t_table_sorted = t_table.sort_values("p-value")
print("Number of significant features =",len(t_table_sorted[t_table_sorted["p-value"] <= 0.05].index))


# In[34]:


#PCA

print('\033[1mPrincipal Component Analysis Plot'.center(100))
df_features = cell_df.drop(['diagnosis','id'], axis=1)
scaled_data = StandardScaler().fit_transform(df_features)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
def diag(x):
    if x =='M':
        return 1
    else:
        return 0
df_diag= cell_df['diagnosis'].apply(diag)
ax = plt.figure(figsize=(12,8))
sns.scatterplot(x_pca[:,0], x_pca[:,1], hue=df['diagnosis'], palette ='Set1' )
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[57]:


#Scree Plot (question 7a)

pca = PCA(n_components=20)
pca.fit(scaled_data)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.xticks(np.arange(0, 18, step=1))
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# In[58]:


#Printing cummulative explained variance ratio

print(np.cumsum(pca.explained_variance_ratio_))


# In[59]:


#Encoding categorical data values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_ML = cell_df 
data_ML['diagnosis'] = le.fit_transform(data_ML['diagnosis'])


# In[60]:


data_ML.head(5)


# In[61]:


# Splitting the dataset into Training and Test set

from sklearn.model_selection import train_test_split
X = cell_df.iloc[:,2:] 
Y = cell_df.iloc[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[62]:


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[63]:


#Fitting the Logistic Regression Algorithm to the Training Set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[64]:


#predicting the Test set results

pred = classifier.predict(X_test)


# In[65]:


print(classification_report(Y_test,pred))


# In[66]:


#Getting the accuracy of the model

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, pred)
print(f' Accuracy : {score * 100}%')


# In[67]:


#Creating the confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


clf = SVC(random_state=0)
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
cm = confusion_matrix(Y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot(cmap = 'GnBu')

plt.show()


# In[ ]:




