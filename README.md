### Intelligent player scouting and talent acquisition for football managers using AI

This work will investigate the use of clustering techniques to identify possible groupings of players based on a blend of various attributes. This approach has resulted in numerous spectacular discoveries that would not have been possible using the manager's manual process.

### Problem statement

- Football managers are typically faced with the difficulty of choosing the right player selections in order to improve their team's performance. This is frequently reflected in their decisions about who to hire (for what position), who to sell, and when to sell.
- Individual players are also difficult to see as long-term investments due to a lack of skill transparency, which prevents management from knowing what has to be developed or optimised in order to maximise a player's worth.
- A lack of data to assess alternatives during player scouting, or a lack of competence to analyse the data itself, can lead to bias and favouritism in decision making.
- Due to the manager's incapacity to recognise overperforming and expensive players, the unsuitable players are commonly selected.
- Manager’s inadequate re-evaluation of players contributes to poor performance of the team; however, a player may perform better in a new position different from the one for which he is known.
- In recent years, football experts and club management have longed for a better means to estimate market value; although crowdsourcing sites such as Transfermarkt have proven useful in estimating market value but this may still be improved with the use of a data-driven approach.


### Proposed solution

- Model can potentially identify patterns that those certain players share in ways that would not normally have been considered by the team managers during their manual evaluation.
- Model would help managers better evaluate their teams to diagnose lack of skill diversity, identify under-valued and over-priced players, and potentially influence their transfer decisions.
- Model support team managers in the selection of players for their teams without human intervention in its decision making
- Models will be able to provide player’s re-evaluation (positions) and recommendations depending on the manager's preferences.
- Understanding of the model may assist the manager in building a suucessfull squad.
- Model can help the football business become more profitable and attract more investment through advertisements, sponsorships and brand ambassadorship.


The structure of the model is described:

The first phase is to effectively build a model capable of grouping players based on their similarity in traits. To do this, I have implemented K-means, K-means++ and DBSCAN algorithms to group players based on their individual abilities, as well as noise removal from the dataset. The model can potentially identify patterns those certain players share in ways that would not normally have been considered by the team managers during their manual evaluation.

The second phase entails building a classification model that will be capable of re-evaluating the player's position and transfer value based on the cluster labels provided by the clustering algorithm in the first phase. These classifiers will be able to predict what group a fresh set of players will belong to. Support Vector Machine and Random Forest are two ML algorithms that I used for this. This would also help managers diagnose lack of skill diversity, identify under-valued and over-priced players, and potentially influence their transfer decisions. The code to the second phase of this work can be found in **Re-evaluation_Classification.ipynb**

### Dataset
FIFA is a football simulation video game in the FIFA series released by Electronic Arts. The football dataset was obtained from Kaggle. The dataset contains 18,207 players, each with a distinct set of attributes. The link to the dataset is provided below. https://www.kaggle.com/karangadiya/fifa19

### Coding and Implementation
I have used Python and its diverse library to build the machine learning pipeline to tackle this classification problem. To improve readability, the code was written in a Jupyter notebook. Python executable code was also provided.
Packages that will be required to run the code:

- Pandas
- Numpy
- Seasborn
- Matplotlib
- Preprocessing/scale
- itertools/product
- OneHotEncoder

Clustering analysis:

- KMeans
- DBSCAN
- PCA
- NearestNeighbors
- Silhouette_score
- Metrics
- Counter
- Plotly/go

Classification analysis:

- RandomForestClassifier
- SVC
- imblearn/SMOTE
- train_test_split
- GridSearchCV
- cross_val_score
- accuracy_score
- precision_score
- recall_score
- f1_score
- jaccard_score
- confusion_matrix

This link will provide additional information about the packages. https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation



