#!/usr/bin/env python
# coding: utf-8

# # Intelligent Player Scouting and Talent Acquisition for Football Managers using AI

# With the use of the FIFA19 dataset, the proposed AI model solves the difficulties managers have while attempting to choose the best players, as well as identifying the average, underperforming, undervalued, and overpriced players.
# 
# 
# **The structure of the model is described in two phases:**
# 
# The first phase is to effectively build a model capable of grouping players based on their similarity in traits. To do this, I have implemented K-means, K-means++ and DBSCAN algorithms to group players based on their individual abilities, as well as noise removal from the dataset. The model can potentially identify patterns those certain players share in ways that would not normally have been considered by the team managers during their manual evaluation.
# 
# The second phase entails building a classification model that will be capable of re-evaluating the players based on the cluster labels provided by the clustering algorithm in the first phase. These classifiers will be able to predict what group a fresh set of players will belong to. Support Vector Machine and Random Forest are two ML algorithms that I used for this. This would also help managers diagnose lack of skill diversity, identify under-priced and over-priced players, and potentially influence their transfer decisions. The code to the second phase of this work can be found in **Re-evaluation_Classification.ipynb**
# 

# ### Importing all of the libraries required to build the model

# In[1]:


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt


# In[2]:


from sklearn.preprocessing import scale
from sklearn import preprocessing
import itertools
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import plotly.graph_objs as go
from itertools import product
from sklearn.neighbors import NearestNeighbors


# ### About the Dataset
# 
# The dataset in use was obtained from Kaggle, which can be accessed online. Please, click on the link to download the dataset. https://www.kaggle.com/karangadiya/fifa19.

# In[5]:


data=pd.read_csv("data.csv")
data.head()


# ### Dataset contains 18,207 rows and 89 columns

# In[6]:


data.shape


# ### Examine the percentage of empty rows also known as NaN (not a number)

# In[7]:


train_test = pd.concat([data.drop('Photo', axis = 1)], keys = ['data'], axis = 0)
missing_values = pd.concat([train_test.isna().sum(),
                            (train_test.isna().sum() / train_test.shape[0]) * 100], axis = 1, 
                           keys = ['Values missing', 'Percent of missing'])
missing_values.loc[missing_values['Percent of missing'] > 0].sort_values(ascending = False, by = 'Percent of missing').style.background_gradient('Blues')


# ### To remove columns that will not be used in the model, I need replace the column identifier ignoring spaces

# In[8]:


data.columns = [c.replace(' ', '') for c in data.columns]
data.columns


# ### Position
# The first way to examining the ground truth is to check the player's position with the greatest number.

# In[9]:


ax = sns.countplot(x = data['Position'])
plt.figure(figsize=(80, 40))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Position') 
ax.set_ylabel('Number of players')
plt.tight_layout()
plt.show()


# ### Age
# Every football player must be considered by their age. It contributes to their market value.

# In[10]:


ax = sns.countplot(x = data['Age'])
plt.figure(figsize=(80, 40))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Age') 
ax.set_ylabel('Number of players')
plt.tight_layout()
plt.show()


# ### Potential 
# Every football player has a unique quality called potential. It describes their expertise, which highly contributes to their market value.

# In[11]:


ax = sns.countplot(x = data['Potential'])
plt.figure(figsize=(80, 40))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Potential') 
ax.set_ylabel('Number of Player')
plt.tight_layout()
plt.show()


# ### Drop any column that aren't necessary for the model.

# In[12]:


data=data.drop(['Name','Unnamed:0','ID','Photo','Flag','Overall','ClubLogo', 'Special', 'InternationalReputation', 'WeakFoot',
               'SkillMoves','WorkRate','BodyType','RealFace','JerseyNumber','Joined','LoanedFrom','ContractValidUntil',
                'Weight','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',
                'LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping',
                'Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure',
                'Marking','SlidingTackle','StandingTackle','ReleaseClause'], axis=1)

data.head(10)


# ### Fill up the empty rows (NaN) with the dataset's mean value
# There are many characteristics that are unimportant for goalkeepers, which is why some of their rows were empty.

# In[13]:


column_means = data.mean()
data = data.fillna(column_means)
data


# ### Remove the Pounds symbol and letters from the players' wages and values.

# In[14]:


data.Wage = data.Wage.str.replace("€","")
data.Wage = data.Wage.str.replace("K","").astype("float")
data.Wage.head() 


# In[15]:


data.Value = data.Value.str.replace("€","")
data.Value = data.Value.str.replace("M","")
data.Value = data.Value.str.replace("K","").astype("float")
data.Value.head() 


# ### One-hot encoding
# This is used to convert all categorical variable into indicator variable i.e., (0's and 1's)

# In[16]:


dummies=pd.get_dummies(data)
dummies


# ### Store the dummy method into variable X

# In[17]:


X=dummies
X


# ### Standardize the dataset
# Now let's normalize the dataset. But why do i need normalization in the first place? Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. I used StandardScaler() to normalize the dataset.

# In[18]:


players_scale = preprocessing.StandardScaler().fit(X).transform(X)
players_scale[0:5]


# In[19]:


#Store the scaled data into a dataframe object
df_players = pd.DataFrame(players_scale, columns=X.columns)
df_players.head()


# # K-Means Algorithm
# 
# ## The following is the analytical strategy used in the K-means experiment:
# 
# ### 1. Applying the elbow method to determine the optimal number of K using the silhouette coefficient
# ### 2. Applying K-means++ to the original dataset 
# ### 3. Hyperparameter tuning for K-means
# ### 4. Applying K-means++ to PCA
# 

# ### 1. Applying the elbow method to determine the optimal number of K using the silhouette coefficient
# inertia: (sum of squared error between each point and its cluster center) as a function of the number of clusters.

# In[20]:


inertia = []
k_list = range(1, 15)

for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(df_players)
    inertia.append([k, km.inertia_])
    
pca_results = pd.DataFrame({'Cluster': range(1,15), 'SSE': inertia})
plt.figure(figsize=(12,6))
plt.plot(pd.DataFrame(inertia)[0], pd.DataFrame(inertia)[1], marker='o', color='green')
plt.title('Optimal Number of Clusters using Elbow Method (Original Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# ### 2. Applying K-means++ to the original dataset
# 
# The plot indicates that the number of clusters should be between 4 and 5, but for the purpose of simplicity, I chose 4 as my preferred number. Compute the sihouette score using k-means++ on the original dataset

# In[21]:


kmeans_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++').fit(df_players)
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(df_players, 
                                                                   kmeans_scale.labels_, metric='euclidean')))
labels_scale = kmeans_scale.labels_
clusters_scale = pd.concat([df_players, pd.DataFrame({'cluster_scaled':labels_scale})], axis=1)


# ### 3. Hyperparameter tuning for K-means

# In[22]:


parameters = {'n_clusters': [2, 3, 4, 5, 10, 20, 30]}

parameter_grid = ParameterGrid(parameters)


# In[23]:


list(parameter_grid)


# In[24]:


best_score = -1
model = KMeans()


# ### 3.1. Fine-tune the K-means model

# In[25]:


for g in parameter_grid:
    model.set_params(**g)
    model.fit(df_players)

    ss = metrics.silhouette_score(df_players, model.labels_)
    print('Parameter: ', g, 'Score: ', ss)
    if ss > best_score:
        best_score = ss
        best_grid = g


# ### 3.2. Get the best silhouette score along with the number of clusters

# In[26]:


best_grid


# ### 3.3. A scatter plot of the original dataset using K-means

# In[44]:


labels_scale=k_means.labels_
pca2 = PCA(n_components=3).fit(df_players)
pca2d = pca2.transform(df_players)
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=labels_scale, 
                palette='Set1',
                s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from Original Dataset', fontsize=15)
plt.legend()
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()


# ### 3.4. Plot a 3-D graph of the original dataset

# In[48]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_scale
trace = go.Scatter3d(x=pca2d[:,0], y=pca2d[:,1], z=pca2d[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# ### 4. Applying K-means++ to PCA 

# In[30]:


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


# ### 4.1. Examine the n components with a value of 2

# In[31]:


pca = PCA(n_components=2)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2'])
print(pca.explained_variance_ratio_)


# ### 4.2. Evaluate the elbow method distribution using PCA (2)

# In[32]:


sse = []
k_list = range(1, 15)

for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(pca_df_scale)
    sse.append([k, km.inertia_])
    
pca_results_scale = pd.DataFrame({'Cluster': range(1,15), 'SSE': sse})
plt.figure(figsize=(12,6))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o', color='green')
plt.title('Optimal Number of Clusters using Elbow Method (PCA_Scaled Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# ### 4.3. After applying PCA (2), recalculate the silhouette score

# In[33]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### 4.4. Examine the n components with a value of 3

# In[34]:


pca = PCA(n_components=3)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2','pc3'])
print(pca.explained_variance_ratio_)


# ### 4.5. Evaluate the elbow method distribution using PCA (3)

# In[35]:


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


# ### 4.6.  After applying PCA (3), recalculate the silhouette score

# In[36]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### 4.7. Examine the n components with a value of 4

# In[37]:


pca = PCA(n_components=4)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2','pc3','pc4'])
print(pca.explained_variance_ratio_)


# ### 4.8. Evaluate the elbow method distribution using PCA (4)

# In[38]:


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


# ### 4.9. After applying PCA (4), recalculate the silhouette score using K-means++

# In[39]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### 4.10. Examine the n components with a value of 5

# In[40]:


pca = PCA(n_components=5)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2','pc3','pc4','pc5'])
print(pca.explained_variance_ratio_)


# ### 4.11. Evaluate the elbow method distribution using PCA (5)

# In[41]:


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


# ### 4.12. After applying PCA (5), recalculate the silhouette score using K-means++

# In[42]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### 4.13. Examine the n components with a value of 30

# In[43]:


pca = PCA(n_components=30)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale)
print(pca.explained_variance_ratio_)


# ### 4.14. Evaluate the elbow method distribution using PCA (30)

# In[44]:


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


# ### 4.15. After applying PCA (30), recalculate the silhouette score using K-means++

# In[45]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# ### 4.16. For the K-means, PCA with a value of 3 produced the best silhouette score.

# In[46]:


pca = PCA(n_components=3)
pca_scale = pca.fit_transform(df_players)
pca_df_scale = pd.DataFrame(pca_scale,  columns=['pc1','pc2','pc3'])
print(pca.explained_variance_ratio_)


# ### 4.17. Evaluate the elbow method distribution using PCA (3)

# In[47]:


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


# ### 4.18. After applying PCA (3), recalculate the silhouette score using K-means++
# I tried a few other numbers for the n component, but it appears that **0.458** is the highest possible score for the silhouette.

# In[48]:


kmeans_pca_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)

print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
labels_pca_scale = kmeans_pca_scale.labels_
clusters_pca_scale = pd.concat([pca_df_scale, 
                                pd.DataFrame({'pca_clusters':labels_pca_scale})], axis=1)


# In[49]:


pca=[{'Number of PCA':2,
         'Number of Clusters': 4,
         'Silhouette Score': 0.411
         },
        {
         'Number of PCA':3,
         'Number of Clusters': 4,
         'Silhouette Score': 0.458
        },
        {'Number of PCA':4,
         'Number of Clusters': 4,
         'Silhouette Score': 0.410
         },
        {'Number of PCA':5,
         'Number of Clusters': 4,
         'Silhouette Score': 0.374
         },
        {'Number of PCA':30,
         'Number of Clusters': 4,
         'Silhouette Score': 0.158
         },]
df=pd.DataFrame(pca, index=['Princpal Component Analysis','Princpal Component Analysis','Princpal Component Analysis','Princpal Component Analysis', 'Princpal Component Analysis'])
df.head()


# ### 4.19. Hyperparameter tunning on K-means after applying PCA (3)

# In[50]:


parameters = {'n_clusters': [2, 3, 4, 5, 10, 20, 30]}

parameter_grid = ParameterGrid(parameters)


# In[51]:


list(parameter_grid)


# In[52]:


best_score = -1
model = KMeans()


# ### 4.20. Fine-tune the model

# In[53]:


for g in parameter_grid:
    model.set_params(**g)
    model.fit(pca_df_scale)

    ss = metrics.silhouette_score(pca_df_scale, model.labels_)
    print('Parameter: ', g, 'Score: ', ss)
    if ss > best_score:
        best_score = ss
        best_grid = g


# ### 4.21. The best silhouette score for K-means++ algorithm is (4) clusters

# In[54]:


best_grid


# ### 4.22. Present a graph that was derived from PCA (3) using K-means++

# In[52]:


plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca_scale.iloc[:,0],
                clusters_pca_scale.iloc[:,1], 
                hue=labels_pca_scale, palette='Set1', s=100, 
                alpha=0.2).set_title('KMeans Clusters (4) Derived from PCA', fontsize=15)
plt.legend()
plt.show()


# ### 4.23. Plot a 3-D graph of K-means clusters derived from PCA (3)

# In[45]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_scale
trace = go.Scatter3d(x=pca2d[:,0], y=pca2d[:,1], z=pca2d[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# # DBSCAN Algorithm
# 
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.
# Dense region, Sparse region, Core point, Border point, Noise point , Density edge , Density connected points
# 
# The DBSCAN algorithm uses two parameters:
# 
# **min points:** The minimum number of points (a threshold) clustered together for a region to be considered dense.
# 
# **epsilon (ε):** A distance measure that will be used to locate the points in the neighborhood of any point.
# 
# The following is the analytical strategy used in the DBSCAN algorithm:
# 
# 
# ### 1. Applying feature extraction to DBSCAN
# ### 2. Construct a 3-D graph to illustrate each cluster distribution (Original Dataset).
# ### 3. Apply elbow method using nearest neighbors
# ### 4. Applying DBSCAN to the original dataset using (eps:0.4; minpts:4)
# ### 5. Hyperparameter tuning for DBSCAN(epsilon & minimum points)
# ### 6. Construct a 3-D graph to illustrate each cluster distribution (original dataset)
# ### 7. Save the results to a file and then analyse them after adjusting the PCA parameters.
# ### 8. Construct a 3-D graph to illustrate each cluster distribution.
# ### 9. Apply a value range of (2,3, & 4) for the PCA n component and examine the results
# ### 10. Eliminate the rows containing noise (-1), store the model label back into a dataframe object
# 
# 

# ### 1. Applying feature extraction to DBSCAN

# In[57]:


pca = PCA(n_components=3)
pca.fit(df_players)
pca_scale = pca.transform(df_players)
pca_df = pd.DataFrame(pca_scale, columns=['pc1', 'pc2', 'pc3'])
print(pca.explained_variance_ratio_)


# ### 2. Construct a 3-D graph to illustrate each cluster distribution (original dataset)

# In[58]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2],
                     mode='markers',marker=dict(colorscale='Greys', opacity=0.3, size = 10, ))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters Derived from Original Data', font=dict(size=12,))
fig.show()


# ### 3. Applying the elbow method using nearest neighbors (2)

# In[64]:


# we use nearestneighbors for calculating distance between points
neigh=NearestNeighbors(n_neighbors=2)
distance=neigh.fit(pca_df)
distances,indices=distance.kneighbors(pca_df)
sorting_distances=np.sort(distances,axis=0)
sorted_distances=sorting_distances[:,1]
plt.figure(figsize=(10,5))
plt.plot(sorted_distances)
plt.xlabel('Distance')
plt.ylabel('Epsilon')
plt.axhline(y=0.4, color='red', ls='--')
plt.grid()
plt.show()


# ### 4. Applying DBSCAN to the original dataset using (eps:0.4; minpts:4)

# In[61]:


dbscan = DBSCAN(eps=0.4, min_samples=4).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 5. Construct a 3-D graph to illustrate each cluster distribution (original dataset)

# In[63]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = db.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (30) Derived from PCA', font=dict(size=12,))
fig.show()


# ### 6. Hyperparameter tuning for DBSCAN (epsilon & minimum points)

# In[65]:


pca_eps_values = np.arange(0.2,2.6,0.1) 
pca_min_samples = np.arange(2,11) 
pca_dbscan_params = list(product(pca_eps_values, pca_min_samples))
pca_no_of_clusters = []
pca_sil_score = []
pca_epsvalues = []
pca_min_samp = []
for p in pca_dbscan_params:
    pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
    pca_epsvalues.append(p[0])
    pca_min_samp.append(p[1])
    pca_no_of_clusters.append(len(np.unique(pca_dbscan_cluster.labels_)))
    pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))
pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
pca_eps_min_df


# ### 7. Save the result to file

# In[66]:


pd.DataFrame(pca_eps_min_df).to_csv('dbscanresultpca1.csv', index=False)


# ### 7.1 I evaluated the obtained result to fine-tune the model

# In[67]:


dbscan = DBSCAN(eps=1.3, min_samples=4).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 8. Construct a 3-D graph to illustrate (4) cluster distributions using the above parameters

# In[1]:


scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = dbscan.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers', marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (4) Derived from PCA', font=dict(size=12,))
fig.show()


# ### 9. Apply a value range of (2,3, & 4) for the PCA n component and examine the results
# Applying PCA (2) to DBSCAN (original dataset)
# 

# In[69]:


pca_dbscan = PCA(n_components=2)
pca_dbscan.fit(df_players)
pca_scale_dbscan = pca_dbscan.transform(df_players)
pca_df = pd.DataFrame(pca_scale_dbscan, columns=['pc1', 'pc2'])
print(pca_dbscan.explained_variance_ratio_)


# ### 9.1. Calculate the epsilon and minimum point parameters while simultaneously eliminating noise to produce the silhouette coefficient

# In[70]:


dbscan = DBSCAN(eps=2, min_samples=2).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 9.2. Hyperparameter tuning for DBSCAN (epsilon & minimum points)

# In[71]:


pca_eps_values = np.arange(0.2,2.1,0.1) 
pca_min_samples = np.arange(2,11) 
pca_dbscan_params = list(product(pca_eps_values, pca_min_samples))
pca_no_of_clusters = []
pca_sil_score = []
pca_epsvalues = []
pca_min_samp = []
for p in pca_dbscan_params:
    pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
    pca_epsvalues.append(p[0])
    pca_min_samp.append(p[1])
    pca_no_of_clusters.append(len(np.unique(pca_dbscan_cluster.labels_)))
    pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))
pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
pca_eps_min_df


# ### 9.3. Save the result to file

# In[72]:


pd.DataFrame(pca_eps_min_df).to_csv('dbscanresultpca2.csv', index=False)


# ### 9.4. I evaluated the obtained result to fine-tune the model

# In[73]:


dbscan = DBSCAN(eps=1.2, min_samples=2).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### Apply a value of 4 for the PCA n component and examine the results

# In[74]:


pca_dbscan = PCA(n_components=4)
pca_dbscan.fit(df_players)
pca_scale_dbscan = pca_dbscan.transform(df_players)
pca_df = pd.DataFrame(pca_scale_dbscan, columns=['pc1', 'pc2','pc3','pc4'])
print(pca_dbscan.explained_variance_ratio_)


# ### 9.5. Construct a 3-D graph to illustrate (4) cluster distributions using the above parameters

# In[2]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = dbscan.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers', marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title='DBSCAN clusters (4) Derived from PCA', font=dict(size=12,))
fig.show()


# ### 9.6. Calculate the epsilon and minimum point parameters while simultaneously eliminating noise to produce the silhouette coefficient

# In[75]:


dbscan = DBSCAN(eps=3.3, min_samples=2).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 9.7 Hyperparameter tuning for DBSCAN (epsilon & minimum points)

# In[76]:


pca_eps_values = np.arange(0.2,3.3,0.1) 
pca_min_samples = np.arange(2,11) 
pca_dbscan_params = list(product(pca_eps_values, pca_min_samples))
pca_no_of_clusters = []
pca_sil_score = []
pca_epsvalues = []
pca_min_samp = []
for p in pca_dbscan_params:
    pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
    pca_epsvalues.append(p[0])
    pca_min_samp.append(p[1])
    pca_no_of_clusters.append(len(np.unique(pca_dbscan_cluster.labels_)))
    pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))
pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
pca_eps_min_df


# ### 9.8. Save the result to file

# In[77]:


pd.DataFrame(pca_eps_min_df).to_csv('dbscanresultpca3.csv', index=False)


# ### 9.9. I evaluated the obtained result to fine-tune the model

# In[78]:


dbscan = DBSCAN(eps=1.7, min_samples=3).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 9.10. Construct a 3-D graph to illustrate (2) cluster distributions using the above parameters

# In[79]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = dbscan.labels_
trace = go.Scatter3d(x=pca_df.iloc[:,0], y=pca_df.iloc[:,1], z=pca_df.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title="'DBSCAN Clusters (4) Derived from PCA'", font=dict(size=12,))
fig.show()


# ### 9.11. For the DBSCAN, PCA with n component value of 2 produced the best silhouette score.

# In[80]:


pca_dbscan = PCA(n_components=2)
pca_dbscan.fit(df_players)
pca_scale_dbscan = pca_dbscan.transform(df_players)
pca_df = pd.DataFrame(pca_scale_dbscan)
print(pca_dbscan.explained_variance_ratio_)


# ### 9.12. Evaluate the obtained result to fine-tune the model

# In[81]:


dbscan = DBSCAN(eps=1.2, min_samples=2).fit(pca_df)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_df, labels))


# ### 10. Eliminate the rows containing noise (-1), store the model label back into a dataframe object

# In[82]:


df_players["Labels"] = labels
df_players.head(10)


# ### 10.1.  Eliminate rows with noise using DataFrame Object

# In[83]:


n_noise_ = list(labels).count(-1)
print('Count:', n_noise_)
indexNames = df_players[df_players['Labels'] == -1 ].index
df_players.drop(indexNames , inplace=True)
df_players.head(10)


# ### 10.2. Confirming the size of the dataset after dropping 5 rows with noise (-1)
# **After removing the noisy datapoint, the dataset has been altered. I'll have to re-import the dataset in order to build a classification model.**

# In[84]:


df_players.shape


# ### 10.3. Examine the labels

# In[85]:


df_players['Labels'].unique()


# In[86]:


clusterNum = 4
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(pca_df_scale)
labels = k_means.labels_
print(labels)


# ### Create a new column for the clustered labels

# In[107]:


df_players["Clusters"] = labels
df_players.head(20)


# # Conclusion
# K-means++ was the method adopted in this research.   At first, one could have assumed that the poor performance was due to the dataset's susceptibility to noise, large dimensionality, or even the cluster shape. The use of PCA on K-means++ has resulted in a more equitable and business-friendly solution. The K-means++ algorithm was able to satisfy the requirements of this research by providing managers with insight on player's skill diversity problems such as underperforming, undervalue, average, overperforming   among many others. The K-means++ method was successful in identifying possible groupings of players based of various attributes. Managers can now understand how the model works and make sound recommendations based on their preferences. The final phase of the project involves re-evaluating the players using a supervised machine learning technique.
# 
# Having said that, I went ahead and used DBSCAN, a density clustering technique commonly employed on non-linear or non-spherical datasets. Two parameters are required: epsilon and minimum points. I also used PCA to reduce the number of dimensions to 3 principal components. I estimated an epsilon value of 0.2 and a minimum point value of 4 using the elbow method. I was able to attain 72 clusters, 1406 noise, and a silhouette score of -0.55 by using this parameter. Admittedly, the findings were unimpressive. To fine-tune the epsilon and minimum points values, I have used an iterative approach. I chose an epsilon value of 1.2 and a minimum points value of 2. The method produced 6 valid clusters, 5 noises, and a silhouette score of 0.46. However, when the generated clusters were plotted, it was observed that the first cluster contained 90% of the players. Similarly, from a business perspective, I would like that the clusters be more evenly distributed in order to give us with useful information about the players. Perhaps DBSCAN is not the best clustering technique for this dataset.

# In[ ]:




