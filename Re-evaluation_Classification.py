#!/usr/bin/env python
# coding: utf-8

# # Classification Model
# This is the second phase of the experiment that involves building a machine learning pipeline using Support Vector Machine and Random Forest. I am going to use the K-means algorithm because the results were fascinating from a business point of view. All of the clusters in the K-means were evenly distributed, providing a strong understanding of the football players.
# 
# DBSCAN results were not even dispersed; over 90% of the football players were concentrated in a single cluster. This would not contribute to the model's core functionality.

# In[8]:


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt


# In[9]:


from sklearn.preprocessing import scale
from sklearn import preprocessing
import itertools
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours 
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[11]:


data=pd.read_csv("data.csv")
data.head()


# ### Examine the size of the dataset to ensure that no changes occurred

# In[12]:


data.shape


# ### Check for NaN (Not a Number)

# In[13]:


data.isnull().sum()


# ### To remove columns that will not be used in the model, I need replace the column identifier ignoring spaces

# In[14]:


data.columns = [c.replace(' ', '') for c in data.columns]
data.columns


# ### Drop any column that aren't necessary for the model

# In[15]:


data=data.drop(['Name','Unnamed:0','ID','Photo','Flag','Overall','ClubLogo', 'Special', 'InternationalReputation', 'WeakFoot',
               'SkillMoves','WorkRate','BodyType','RealFace','JerseyNumber','Joined','LoanedFrom','ContractValidUntil',
                'Weight','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',
                'LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping',
                'Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure',
                'Marking','SlidingTackle','StandingTackle','ReleaseClause'], axis=1)

data.head(10)


# ### Fill up the empty rows (NaN) with the dataset's mean value
# 

# In[16]:


column_means = data.mean()
data = data.fillna(column_means)
data


# ### Remove the Pounds symbol and letters from the players' wages and values

# In[17]:


data.Wage = data.Wage.str.replace("€","")
data.Wage = data.Wage.str.replace("K","").astype("float")
data.Wage.head() 


# In[ ]:


data.Value = data.Value.str.replace("€","")
data.Value = data.Value.str.replace("M","")
data.Value = data.Value.str.replace("K","").astype("float")
data.Value.head() 


# ### One-hot encoding
# This is used to convert all categorical variable into indicator variable i.e., (0's and 1's)

# In[18]:


dummies=pd.get_dummies(data)
dummies


# ### Store the dummy method into variable X

# In[19]:


X=dummies
X


# ### Normalize the dataset

# In[20]:


players_scale = preprocessing.StandardScaler().fit(X).transform(X)
players_scale[0:5]


# In[21]:


df_players = pd.DataFrame(players_scale, columns=X.columns)
df_players.head()


# ### Applying K-means++ to PCA principal components

# In[22]:


#n_components=900 because we have 900 features in the dataset
pca = PCA(n_components=900)
pca.fit(df_players)
variance = pca.explained_variance_ratio_
var = np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12,6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(0,100.5)
plt.plot(var)


# ### Examine the n components with a value of 3 given that it produced the best silhouette score previously.

# In[23]:


pca = PCA(n_components=3)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2','pc3'])
print(pca.explained_variance_ratio_)


# ### Applying silhouette coefficient (using the elbow method)

# In[24]:


sse = []
k_list = range(1, 15)

for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(pca_df_scale)
    sse.append([k, km.inertia_])
    
pca_results = pd.DataFrame({'Cluster': range(1,15), 'SSE': sse})
plt.figure(figsize=(12,6))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o', color='green')
plt.title('Optimal Number of Clusters using Elbow Method (PCA_Scaled Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# ### Applying K-means++ to PCA and finding out the optimal number of clusters
# Recall that PCA component of 3 gave us the best silhouette score

# In[25]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### Execute the K-means++ model

# In[26]:


clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(pca_df_scale)
labels = k_means.labels_
print(labels)


# ### Create a new column for the clustered labels

# In[27]:


df_players["Clusters"] = labels
df_players.head(20)


# ### Examine the size of the dataset

# In[28]:


df_players.shape


# ### Separate the data into features (X) and targets (clusters (y)).

# In[29]:


X = df_players.drop('Clusters', axis=1)
y = df_players['Clusters']


# ### Over-sampling and under-sampling on unbalanced data

# In[30]:


print(imblearn.__version__)

oversample = SMOTE()
enn = EditedNearestNeighbours()
# label encode the target variable

y = LabelEncoder().fit_transform(y)

X, y = enn.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100

    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# ### Train Test Split divides the dataset into 70 percent for training and 30 percent for testing.

# In[31]:


x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=50)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# ### Insights on the clustering pattern

# In[32]:


ax = sns.countplot(x = df_players['Clusters'])
plt.figure(figsize=(80, 40))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Clusters') 
ax.set_ylabel('Number of players')
plt.tight_layout()
#plt.title("Visualization of players based on their position")
plt.show()


# ### Support Vector Machine
# To build this model, I have used the Support Vector Machine Classifier

# In[33]:


clf = SVC()


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy.

# In[34]:


# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(x_train, y_train)


# ### This will produce the best parameters, estimator, and score for the SVM classifier

# In[35]:


print('Best parameter:',grid.best_params_)
print('Grid best estimator:',grid.best_estimator_)
print('Best score:',grid.best_score_)


# ### Will now, apply the above-mentioned parameters for the SVM classifier

# In[36]:


clf = SVC(C=1, gamma=0.0001, kernel= 'rbf')
clf.fit(x_train, y_train) 


# ### Will apply the predict method to the test set

# In[37]:


y_pred = clf.predict(x_test)


# ### I used cross validation to further analyze the model's performance on its test set

# In[38]:


print(classification_report(y_test, y_pred))

print('Accuracy of SVM classifier on the training set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of SVM classifier on the test set: {:.2f}'.format(clf.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(clf, x_train, y_train, cv=10) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: ',score.mean())


# ### Summarize the model's performance using different classification metrics

# In[39]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)
    
summarize_classification(y_test, y_pred)


# ### To further evaluate our performance findings, let's build a confusion matrix that describes the false positive, true negative, true positive, and false negative

# In[40]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[41]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False(0)','True(1)'],normalize= False,  title='Confusion matrix')


# ### Summary of the confusion matrix
# Looking at the first row. The first row contains players whose false value in the test set is (0). As you can see, 365 of the 712 players have a false value of (0). And, of these 365, the classifier accurately predicted 365 as (0), and 0 as (1) (True value) for the predicted labels.
# 
# This indicates that in the test set, the actual false value for 365 players was (0), and the classifier accurately predicted those as (0) (True label). However, while the actual label of 1 player was 0 (False value), the classifier predicted those as 0, which means it did excellently well. We may think of it as a model excellence for the first row.
# 
# What about the players that have a true value of 1?
# 
# Looking at the second row. It appears that there are 356 players whose true value was 1. The classifier accurately identified 356 of them as 1 as a result. it has done an excellent job at predicting players with true value 1. The confusion matrix is useful since it demonstrates the model's ability to properly predict or separate the classes. In the case of a binary classifier, such as this one, these values can be interpreted as the number of true positives, false positives, true negatives, and false negatives.

# ### Comparison of the actual test set to the predicted labels

# In[42]:


#Accuracy
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(10)


# # Random Forest

# In[43]:


rf=RandomForestClassifier()
rf


# ### Grid search cross validation hyperparameter tuning will be used to improve our model's performance accuracy

# In[44]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# In[45]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)


# In[46]:


grid_search.fit(x_train,y_train)


# ### This will reccommend the best parameters, estimator, and score for the Random Forest classifier

# In[47]:


print('Best parameter:', grid_search.best_params_)
print('Best grid estimator:', grid_search.best_estimator_)
print('Best score', grid_search.best_score_)


# ### I applied the above-mentioned parameters for the Random Forest classifier

# In[48]:


rf=RandomForestClassifier(bootstrap=True, max_depth=80, max_features=3, 
                          min_samples_leaf=3, min_samples_split=8,
                          n_estimators=200).fit(x_train,y_train)
rf


# ### I applied the predict method to the test set

# In[49]:


y_pred = rf.predict(x_test)


# ### I used cross validation to further analyze the model's performance on its test set

# In[50]:


print(classification_report(y_test, y_pred))

print('Accuracy of Random Forest classifier on the training set: {:.2f}'.format(rf.score(x_train, y_train)))
print('Accuracy of Random Forest classifier on the test set: {:.2f}'.format(rf.score(x_test, y_test)))

#Decision Trees are very prone to overfitting as shown in the scores

score = cross_val_score(rf, x_train, y_train, cv=5) 
print('Cross-validation score: ',score)
print('Cross-validation mean score: ',score.mean())


# ### Summarize the model's performance using different classification metrics

# In[51]:


def summarize_classification(y_test,y_pred,avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred,normalize=True)
    num_acc = accuracy_score(y_test, y_pred,normalize=False)
    f1= f1_score(y_test, y_pred, average=avg_method)
    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)
    jaccard = jaccard_score(y_test, y_pred, average=avg_method)
    
    print("Length of testing data: ", len(y_test))
    print("accuracy_count : " , num_acc)
    print("accuracy_score : " , acc)
    print("f1_score : " , f1)
    print("precision_score : " , prec)
    print("recall_score : ", recall)
    print("jaccard_score : ", jaccard)
    
summarize_classification(y_test, y_pred)


# ### Comparison of the actual test set to the predicted labels

# In[52]:


#Accuracy
pred_results = pd.DataFrame({'y_test': pd.Series(y_test),
                             'y_pred': pd.Series(y_pred)})

pred_results.sample(10)


# ### To further evaluate our performance findings, let's build a confusion matrix that describes the false positive, true negative, true positive, and false negative.

# In[53]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[54]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['False(0)','True(1)'],normalize= False,  title='Confusion matrix')


# ### Summary of the confusion matrix
# Looking at the first row. The first row contains players whose false value in the test set is (0). As you can see, 322 of the 671 players have a false value of (0). And, of these 322, the classifier accurately predicted 322 as (0), and 0 as (1) for the predicted label.
# 
# This indicates that in the test set, the actual false value for 322 players was (0), and the classifier accurately predicted those as (0). However, while the actual label of 0 players was 0 (false value), the classifier predicted those as 0, which means it did excellently well. We may think of it as a model excellence for the first row.
# 
# What about the players that have a true value of 1?
# 
# Looking at the second row. It appears that there are 347 players whose true value was 1. The classifier accurately identified 347 of them as 1, and 2 of them wrongly as 0 (false). As a result, it has done a good job at predicting players with true value 1.

# ### Final review of the results and evaluations
# This provides an understanding of the business perspective related to our model. The clustering method (K-means) was able to categorise players according to their attributes. We may also conclude that the algorithm accurately identified the average, undervalued, and overperforming players. It also produced astounding results for the re-evaluation of the players, with 99 percent accuracy on the test set.

# In[55]:


data["Clusters"] = labels
data.head(20)


# ### Model should be saved to file for adequate evaluation.

# In[56]:


pd.DataFrame(data).to_csv('playerclusters.csv', index=False)


# ### Evaluation metric results

# In[58]:


result=[{ 'Accuracy Score':'99%',
         'F1 Score': '99%',
         'Precision Score': '99%',
         'Recall Score': '99%',
         'Jaccard Score': '99%'},
        {'Accuracy Score':'96%',
         'F1 Score': '96%',
         'Precision Score': '96%',
         'Recall Score': '96%',
         'Jaccard Score': '93%'}]
df=pd.DataFrame(result, index=['Support Vector Machine','Random Forest'])
df.head()


# # Conclusion
# On the test set, the classification model generated excellent results. Because I have adequate data to train on, the model is not prone to overfitting. On the test set, Support Vector Machine and Random Forest achieve 98 and 99 percent accuracy, respectively. The f1 score and recall for all classes yielded 100 percent score for SVM classifier. This will address the manager's re-evaluation problem by predicting which category a given player’s skill set should belong. Managers can now determine if a player's release clause is genuinely worth the amount asked on the transfer market and which players should be rotated into other positions. This model has helped managers in diagnosing a lack of skill diversity and potentially influencing transfer decisions. Models now offers recommendation on players based on the manager's preferences.
# 
# K-means++ was the method adopted in this research. At first, one could have assumed that the poor performance was due to the dataset's susceptibility to noise, large dimensionality, or even the cluster shape. The use of PCA on K-means++ has resulted in a more equitable and business-friendly solution. The K-means++ algorithm was able to satisfy the requirements of this research by providing managers with insight on player's skill diversity problems such as underperforming, undervalue, average, overperforming among many others. The K-means++ method was successful in identifying possible groupings of players based of various attributes. Managers can now understand how the model works and make sound recommendations based on their preferences. The final phase of the project involves re-evaluating the players using a supervised machine learning technique. 
# 
# Having said that, I went ahead and used DBSCAN, a density clustering technique commonly employed on non-linear or non-spherical datasets. Two parameters are required: epsilon and minimum points. I also used PCA to reduce the number of dimensions to 3 principal components. I estimated an epsilon value of 0.2 and a minimum point value of 4 using the elbow method. I was able to attain 72 clusters, 1406 noise, and a silhouette score of -0.55 by using this parameter. Admittedly, the findings were unimpressive. To fine-tune the epsilon and minimum points values, I have used an iterative approach. I chose an epsilon value of 1.2 and a minimum points value of 2. The method produced 6 valid clusters, 5 noises, and a silhouette score of 0.46. However, when the generated clusters were plotted, it was observed that the first cluster contained 90% of the players. Similarly, from a business perspective, I would like that the clusters be more evenly distributed in order to give us with useful information about the players. Perhaps DBSCAN is not the best clustering technique for this dataset.
# 
# 

# In[ ]:




