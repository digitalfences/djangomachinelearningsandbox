# djangomachinelearningsandbox

A sandbox project to practice implementing machine learning in django projects. Currently the app is passed a sound string and decides whether or not the creature that made the sound is a dog or not. 

## How To Use

I followed along with a short tutorial that describes this app
https://towardsdatascience.com/productionize-a-machine-learning-model-with-a-django-api-c774cb47698c

To use this project in your local env, make sure to follow the steps for setting up the env folder. You will also need a valid django secret key for the project. The one I used for this project is on my local repo for security reasons, you will either need to generate a new one or message me for the project key I used. 

After setting up your local repo, you will be able to run the project by changing into the root directory and then the api directory. This is the django project, which manages the predictor app. The app is a simple linear regression on the data I wrote into the jupyter notebook as outlined here

```python
data = [
    ['woof', 1],
    ['bark', 1],
    ['ruff', 1],
    ['bowwow', 1],
    ['roar', 0],
    ['bah', 0],
    ['meow', 0],
    ['ribbit', 0],
    ['moo', 0],
    ['yip', 0],
    ['pika', 0]
]
X = []
y = []
for i in data:
    X.append( i[0] )
    y.append( i[1] )
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
from sklearn.linear_model import LinearRegression
import numpy as np
regressor = LinearRegression()
regressor.fit(X_vectorized, y)
test_feature = vectorizer.transform(['woof'])
prediction = regressor.predict(test_feature)
print(prediction)
test_feature = vectorizer.transform(['ribbit'])
prediction = regressor.predict(test_feature)
print(prediction)
test_feature = vectorizer.transform(['meoww'])
prediction = regressor.predict(test_feature)
print(prediction)

import pickle
pickl = {
    'vectorizer': vectorizer,
    'regressor': regressor
}
pickle.dump( pickl, open( 'models' + ".p", "wb" ) )
```
This uses some powerful python libraries and the dataset I added to identify which sounds are likely to correspond to a dog versus a not-a-dog. It is simple, but I plan to add complexity to this app to analyze multiple variables beside sound, and instead of deciding between a dog or not-a-dog, I plan to use unsupervised learning to allow the app to draw its own categories of animals in 2.0.
