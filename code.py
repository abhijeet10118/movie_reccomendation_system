"""
Movie recommender system with TensorFlow Recommenders (TFRS) on the Netflix dataset.

A two-tower, multi-task model:
  * Retrieval task  -> recommends movies for a user
  * Ranking task    -> predicts the rating a user would give a movie

Usage:
    python netflix_movie_recommender.py
"""

import random
from typing import Dict, Text

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RATINGS_PATH = "../netflix-project/data/Netflix_Dataset_Rating.csv"
MOVIES_PATH = "../netflix-project/data/Netflix_Dataset_Movie.csv"

SEED = 42
EMBEDDING_DIMENSION = 32
TRAIN_SIZE = 80_000
TEST_SIZE = 20_000
SHUFFLE_BUFFER = 100_000
TRAIN_BATCH_SIZE = 8_192
TEST_BATCH_SIZE = 4_096
EPOCHS = 3
RATING_WEIGHT = 0.9
RETRIEVAL_WEIGHT = 1.0
SHOW_PLOTS = True  # set to False for headless runs


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data():
    ratings_df = pd.read_csv(RATINGS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)

    # merge movie metadata (Year, Name) into the ratings
    ratings_df = ratings_df.merge(movies_df, on="Movie_ID")

    print(ratings_df.head())
    ratings_df.info()
    return ratings_df, movies_df


# --------------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------------- #
def run_eda(ratings_df: pd.DataFrame) -> None:
    ratings_per_user = ratings_df.groupby("User_ID")["Rating"].count()
    print(f"On average, each user has rated {int(ratings_per_user.mean())} movies in the dataset")
    print(ratings_per_user.sort_values(ascending=False).head())

    if not SHOW_PLOTS:
        return

    # Movies released per year
    yearly_movie_counts = ratings_df.groupby("Year")["Name"].nunique()
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_movie_counts.index, yearly_movie_counts.values, color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Total Number of Movies Released")
    plt.title("Total Number of Movies Released Each Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Rating distribution
    rating_counts = ratings_df["Rating"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        rating_counts,
        labels=rating_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=["skyblue", "lightgreen", "gold", "lightcoral", "lightsalmon"],
    )
    plt.title("Percentage of Each Rating Value Given by Users")
    plt.axis("equal")
    plt.show()

    # Year released vs. rating (no correlation expected)
    plt.figure(figsize=(10, 6))
    plt.scatter(ratings_df["Year"], ratings_df["Rating"], color="blue", alpha=0.5)
    plt.xlabel("Year Released")
    plt.ylabel("Rating")
    plt.title("Influence of Year Released on Rating")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------------------- #
# Data processing
# --------------------------------------------------------------------------- #
def build_datasets(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    # User IDs must be strings for the StringLookup layer
    ratings_df["User_ID"] = ratings_df["User_ID"].astype("str")

    ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[["User_ID", "Rating", "Name"]]))
    movies = tf.data.Dataset.from_tensor_slices(dict(movies_df[["Name"]]))

    ratings = ratings.map(
        lambda x: {"Name": x["Name"], "User_ID": x["User_ID"], "Rating": x["Rating"]}
    )
    movies = movies.map(lambda x: x["Name"])
    print(f"Total Data: {len(ratings)}")

    # Train / test split
    tf.random.set_seed(SEED)
    shuffled = ratings.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=False)
    train = shuffled.take(TRAIN_SIZE)
    test = shuffled.skip(TRAIN_SIZE).take(TEST_SIZE)

    # Vocabularies
    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["User_ID"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    print(f"Unique movie titles: {len(unique_movie_titles)}")
    print(f"Unique user ids: {len(unique_user_ids)}")
    print(unique_movie_titles[:10])

    return ratings, movies, train, test, unique_movie_titles, unique_user_ids


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MovieModel(tfrs.models.Model):
    """Two-tower multi-task model (rating + retrieval)."""

    def __init__(
        self,
        rating_weight: float,
        retrieval_weight: float,
        movies: tf.data.Dataset,
        unique_movie_titles: np.ndarray,
        unique_user_ids: np.ndarray,
        embedding_dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        super().__init__()

        # Movie tower
        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            # +1 to account for unknown tokens
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension),
        ])

        # User tower
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        ])

        # Small network that predicts a rating from concatenated embeddings
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # Tasks
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # Loss weights
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]):
        user_embeddings = self.user_model(features["User_ID"])
        movie_embeddings = self.movie_model(features["Name"])

        return (
            user_embeddings,
            movie_embeddings,
            self.rating_model(tf.concat([user_embeddings, movie_embeddings], axis=1)),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        ratings = features.pop("Rating")
        user_embeddings, movie_embeddings, rating_predictions = self(features)

        rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def make_optimizer():
    # The legacy optimizer is used in the original notebook; fall back if unavailable.
    try:
        return tf.keras.optimizers.legacy.Adagrad(0.1)
    except AttributeError:
        return tf.keras.optimizers.Adagrad(0.1)


def train_and_evaluate(model, train, test):
    model.compile(optimizer=make_optimizer())

    cached_train = train.shuffle(SHUFFLE_BUFFER).batch(TRAIN_BATCH_SIZE).cache()
    cached_test = test.batch(TEST_BATCH_SIZE).cache()

    model.fit(cached_train, epochs=EPOCHS)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}")
    return cached_train, cached_test


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def build_index(model, movies):
    """Build a BruteForce retrieval index (swap for ScaNN on large catalogs)."""
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
    )
    return index


def predict_movie(index, user, top_n=5):
    _, titles = index(tf.constant([str(user)]))

    print(f"Top {top_n} recommendations for user {user}:\n")

    unique_titles = set()
    for title in titles[0].numpy():
        title_str = title.decode("utf-8")
        if title_str not in unique_titles:
            unique_titles.add(title_str)
            print(f"{len(unique_titles)}. {title_str}")
            if len(unique_titles) == top_n:
                break


def predict_rating(model, user, movie):
    _, _, predicted_rating = model({
        "User_ID": np.array([str(user)]),
        "Name": np.array([movie]),
    })
    print(f"Predicted rating for {movie}: {predicted_rating.numpy()}")


def pick_random_test_user(cached_test) -> str:
    """Pick a random User_ID from a random batch of the test set."""
    batches = list(cached_test)
    batch = random.choice(batches)
    idx = random.randint(0, len(batch["User_ID"]) - 1)
    return batch["User_ID"][idx].numpy().decode("utf-8")


def user_in_train(cached_train, user_id: str) -> bool:
    target = user_id.encode("utf-8")
    for batch in cached_train:
        if target in batch["User_ID"].numpy():
            return True
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    random.seed(SEED)

    ratings_df, movies_df = load_data()
    run_eda(ratings_df)

    ratings, movies, train, test, unique_movie_titles, unique_user_ids = build_datasets(
        ratings_df, movies_df
    )

    model = MovieModel(
        rating_weight=RATING_WEIGHT,
        retrieval_weight=RETRIEVAL_WEIGHT,
        movies=movies,
        unique_movie_titles=unique_movie_titles,
        unique_user_ids=unique_user_ids,
    )

    cached_train, cached_test = train_and_evaluate(model, train, test)

    # Pick a random user from the test set and check whether they were seen in training
    random_user = pick_random_test_user(cached_test)
    print(f"Randomly selected 'User_ID': {random_user}")
    if user_in_train(cached_train, random_user):
        print(f"User {random_user} exists in the training dataset.")
    else:
        print(f"User {random_user} does not exist in the training dataset.")

    # Recommendations for a specific user
    user_id = "169999"
    index = build_index(model, movies)
    predict_movie(index, user_id, top_n=10)
    predict_rating(model, user_id, b"Pride and Prejudice")

    # Show the user's rating history for comparison
    history = ratings_df[ratings_df["User_ID"] == user_id]
    print(history.head(20))


if __name__ == "__main__":
    main()