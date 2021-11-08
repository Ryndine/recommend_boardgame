# Boardgame Recommender System
[Ryan Mangeno](https://github.com/Ryndine) | [Anny Tritchler](https://github.com/tritchlin/) | [Melissa Cardenas](https://github.com/melcardenas28)

## Objective: 
The goal for this project is to implement everything we learned in our machine learning course into one final project within a week's time. After deliberation we settle on a boardgame recommender system. Reasoning because there are often far too many boardgames that exist which we do not know about, and recommendations are almost always through word of mouth. I personally do not know of any good recommendation systems out for this small industry.

## Tools & databases used:
Python, Jupyter, Pandas, SQLite, Matplotlib, Scikit-Learn, Tensorflow, KMeans, KNearestNeighbor, Keras, Cosine Simularities.

## Resources:
- [Kaggle 20,000 Boardgames Database](https://www.kaggle.com/extralime/20000-boardgames-dataset?select=boardgames1.csv)

## Part 1: Project methods and directions:
 
**Data Collection**

We found many databases already created that used various craping methods. However many were missive important fields to create dummies for, or fields for running an effective recommender on. After finding our database with 20,000 boardgames we saw it had more than enough scraped data to go off of, however it did not provide us with any user data. Because of this reason we would be limited to the type of recommendation we would be able to provide.

**Database Cleanup**

Cleaning the database was simple, but with minor annoyances. First was the database came with unicode in the gameboard names, which is solved with a simple unicode decoder:
```
# Our decoding
decode_lambda = lambda x: bytearray(x, 'utf-8').decode('unicode-escape')
# Applying the decoding to the column, ignore errors.
boardgame_df['name'] = boardgame_df['name'].apply(lambda x: decode_lambda(x))
```
After that the column names were cleaned up a little better for visual appeal.
```
boardgame_df = boardgame_df.rename(columns={'objectid': 'ID', 'name': 'Name', 'average': 'Avg Rating', 'avgweight': 'Complexity', 'boardgamecategory': 'Category', 'boardgamemechanic': 'Mechanic'})
```

**Machine Learning Preperation**

In order to build a recommender, we needed to break out all the categories and mechanics in order to create dummies which had a binary input to show whether or not a given boardgame had that cotegorization or not. However the columns for these were discovered to be string values, instead of lists, which they appeared to be. To clean this up we had to trip the brackets off the strings, then split on the comma:
```
categories_df = boardgame_df[['ID', 'Category']].copy()
categories_df['Category'] = categories_df['Category'].apply(lambda x: x[1:len(x)-1].split(', '))
```
After converting the string into a list of strings, we were then able to continue forward with exploding the list out for individual categories in order to find how many boardgames were in each.
```
cat_counts = categories_df.explode('Category')
cat_counts.columns.str.replace("'","")
cat_vc = cat_counts.groupby("Category").size().sort_values(ascending=False)
```
From here in order to improve the accuracy of the training and testing stages we created cutoff point for categories with less than 50 boardgames. We put these into a new "other" category.
```
categories_to_replace = cat_vc[cat_vc < 50].index
# Replace in dataframe
for cats in categories_to_replace:
    cat_counts['Category'] = cat_counts['Category'].replace(cats,"Other")
```
After that we were able to take all our categories and create a binary dataframe using get_dummies().
```
cat_dummies = pd.get_dummies(cat_counts)
cat_final = cat_dummies.groupby("ID").sum()
```
This entire process was repeated for the boardgame mechanics. Then with the newly cleaned and prepared dataframes we simply merged everything together.
```
dfs_to_merge = [boardgame_trunc, cat_final, mech_final]
boardgame_inter = pd.merge(boardgame_trunc, cat_final, on="ID", how='outer')
boardgame_final = pd.merge(boardgame_inter, mech_final, on="ID", how='outer')
```

**Clustering with KMeans**

At this point were having some concerns with the project. We realized, we actually could use what we currently had to create a recommendation system without any machine learning at all. So what was the purpose of machine learning? We were using machine learning algorithms to solve what we needed, but using an algorithm wasn't machine learning. Going forward we focused on putting the database we had through the machine learning process to see how well it could predict a boardgame with the given data.

Since we did not have any data to train machine learning on, we had to go with clustering the data we had. To do this we used KMeans, and since we were trying to predict boardgames, we dropped the column "Name". Then we created an eblow graph in order to observe how many clusters we likely needed for our dataset.
```
ml_df = boardgame_final.drop(columns='Name')
inertia = []
k = list(range(1, 11))
# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=5)
    km.fit(ml_df)
    inertia.append(km.inertia_)
# Create the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
```
![elbow_graph](https://github.com/Ryndine/recommend_boardgame/blob/main/Images/elbow.jpeg)

We decided the best clustering for our data was 4, so we ran a simple script to cluster the data.
```
def get_clusters(k, data):
    # Initialize the K-Means model
    model = KMeans(n_clusters=k, random_state=0)
    # Train the model
    model.fit(data)
    # Create return DataFrame with predicted clusters
    data["class"] = model.labels_
    return data
clusters = get_clusters(4, ml_df)
```
![clusters_graph](https://github.com/Ryndine/recommend_boardgame/blob/main/Images/clusters.jpeg)

Unfortunately our plotting of the clusters doesn't reveal anything meaningful, so we moved forward with training and testing.

**K Nearest Neighbor Accuracy**

To start the KNN treain & test, we dropped "ID" from the database since we're trying to find how accurate predicting boardgames would be.
```
y = clusters['ID']
X = clusters.drop(columns='ID')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
```
We create our X and Y train and test variables and fir the training to the KNN algorithm.
```
knn.score(X_train, y_train)
knn.score(X_test, y_test)
```
Unfortunately our scores for KNN returned results of .17 for train, and .0 for test. We tried cleaning the data further, adjusting clusters, and cuttoff points for the categories, and different variables. However, the results never went beyond .20 accuracy.

**Keras Accuracy**

Due to KNN yielding unsatisfactory results, we looking towards neural networks for a different approach. We ran the same train and test variables into Keras and started by preparing the variables and hyperparameters. 
```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)
```
For our hyperparameters we're using a relu layer and a softmax layer.
```
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(100, input_dim=784, activation="relu"))
nn_model.add(tf.keras.layers.Dense(10, activation="softmax"))
nn_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
```
Then fitting the model to check the accuracy.
```
fit_model = nn_model.fit(x_train, y_train, batch_size=200, epochs=20, verbose=1)
model_loss, model_accuracy = nn_model.evaluate(x_train,y_train,verbose=1)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
![accuracy_graph](https://github.com/Ryndine/recommend_boardgame/blob/main/Images/accuracy.jpeg)

The final score being a Loss of .26 and Accuracy of .92. Unfortunately myself and the team don't exactly understand why the score improved so much. These results are something we plan on looking into far more now that the project is over.

## Part 2: Recommendation System with Collaborative Filtering

For now, I wanted to return to the project and create a collaborative filter based recommendation system, seen [here](https://github.com/Ryndine/recommend_boardgame/blob/main/collab_recommend.ipynb).

To do this I'm starting by taking my cleaned dataframe and dropping "Ratings" and "Complexity". I do not want to focus on these variables for this recommender.
```
data = boardgame_final.drop(['Avg Rating', 'Complexity'], axis=1)
# Create a new dataframe without the user ids.
data_items = data.drop('ID', axis=1)
data_items = data_items.set_index('Name')
data_items.head()
```
Next I needed to further prepare the database. I wanted to have my boardgames as columns and my categories as rows. To do this I transposed the dataframe, create a new index for the category rows, clean up for type errors, then create two separate dataframes for later.
```
data_items = pd.DataFrame.transpose(data_items)
data_items.reset_index(level=0, inplace=True)
data_items = data_items.rename(columns={'index': 'Name'})
data_items.columns.names = ['']
# Save save current dataframe for reading category names.
data = data_items
# Drop name to create a second DF with only integers.
data_items = data_items.drop('Name', axis=1)
data_items.head()
```
Moving forward I now have a Dataframe with Names, and a dataframe that is completely binary. Using the binary dataframe I can calculate simularity between boardgames using the cosine simularity from sklearn.
```
# BG-BG CALCULATIONS

magnitude = np.sqrt(np.square(data_items).sum(axis=1))

data_items = data_items.divide(magnitude, axis='index')

def calculate_similarity(data_items):
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim

data_matrix = calculate_similarity(data_items)
```
```
print(data_matrix.loc['Gloomhaven'].nlargest(11))
```
![test](https://github.com/Ryndine/recommend_boardgame/blob/main/Images/similarity_test.jpg)

With positive results from boardgame to boardgame calculations I'm confident in moving forward with category to boardgame calculations. I'm doing this step in place of "user to item" since my dataset does not have user data. However the results and operations are similar.
```
# CATEGORY-BOARDGAME CALCULATIONS - NO NEIGHBORS

category = 'Adventure' # The id of the user for whom we want to generate recommendations
category_index = data[data.Name == category].index.tolist()[0] # Get the frame index

# Get the boardgames in selected category.
category_boardgames = data_items.iloc[category_index]
category_boardgames = category_boardgames[category_boardgames >0].index.values

# Boardgames for all items as a sparse vector.
user_rating_vector = data_items.iloc[category_index]

# Calculate the score.
score = data_matrix.dot(user_rating_vector).div(data_matrix.sum(axis=1))

# Remove the known boardgames from the recommendation.
score = score.drop(category_boardgames)
```
```
print(category_boardgames)
print(score.nlargest(20))
```
![cat-bg-noneighbor](https://github.com/Ryndine/recommend_boardgame/blob/main/Images/cat_bg_test.jpg)

As you see from the results here the simularity between games is very low. The first print shows boardgames that share the category "Adventure" but we're removing from our recommendation. The results given from the score are all boardgames which are not inside the category "Adventure" but share simular categories as "Gloomhaven". When used with a user recommendation system, this would make it so a user would receive new recommendations that they haven't liked or disliked yet. 

The only thing left to add to this system is neighbors in order to improve results, which I'm currently working on finishing.
