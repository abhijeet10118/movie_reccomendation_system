# Movie recommenders with TensorFlow Recommenders using Netflix Dataset

recommender system using TensorFlow Recommenders (TFRS)

The primary goal of a recommender system is to to anticipate the behavioral tendencies, likes, or dislikes of users and offer individualized suggestions for items they are most likely to engage with.

### Imports
```
!pip install -q tensorflow-recommenders
```

### The dataset
For this project, we will be using the Netflix Movie Dataset, available <a href="https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv%5C">here</a>

This recommender will use both implicit (movie watches) and explicit signals(ratings) 

The walkthrough of this project is available <a href="https://www.joankusuma.com/post/project-movie-recommender-system-with-tensorflow-recommenders-and-netflix-dataset">here</a>

### The model will be built on a two-tower model with two tasks: retrieval and ranking:
<img src="https://static.wixstatic.com/media/81114d_67d2be126a4843e19e0ef31d5705aaeb~mv2.png" alt="model-architecture" height="300" width="500">

The two-tower model will include the following:
* User-tower: turns 'User_ID's into user-embeddings (high-dimensional vector representations)
* Movie-tower: turns movie titles ('Name') into movie-embeddings

This model will have 2 tasks:
* Rating (Ranking): MSE (loss to predict ratings), RMSE (metrics)
* Retrieval: the retrieval task object is a wrapper that bundles together the loss function and metric computation. Top-K metric will be used
