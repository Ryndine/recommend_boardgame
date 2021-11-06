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

**K Nearest Neighbor Accuracy**

**Keras Accuracy**

## Part 2: Recommendation System with Collborative Filtering
