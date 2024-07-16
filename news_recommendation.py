from typing import Dict, Text

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow_recommenders as tfrs

import string
import re
import sklearn
import seaborn as sns
from collections import Counter
from ast import literal_eval
from datetime import datetime
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlalchemy
import pymysql
import warnings
warnings.filterwarnings('ignore')

engine = sqlalchemy.create_engine('mysql+pymysql://User:Password@Hostname:3306/DB_name')

rating = pd.read_sql_table('rating', engine)
article = pd.read_sql_table('article', engine)

df = pd.DataFrame(article, columns=['Title', 'Url'])

article.head()

rating.head()

print("Banyaknya baris article sebelum preprocesssing =", len(article))
print("Banyaknya baris rating sebelum preprocesssing =", len(rating))

article = article.dropna()
#rating = rating.dropna()

article = article.drop_duplicates(subset=None, keep='first', inplace=False)
#rating = rating.drop_duplicates(subset=None, keep='first', inplace=False)

print("Banyaknya baris article setelah preprocesssing =", len(article))
print("Banyaknya baris rating setelah preprocesssing =", len(rating))

rating = rating.astype(np.str)
article = article.astype(np.str)

ratings = tf.data.Dataset.from_tensor_slices(dict(rating))
articles = tf.data.Dataset.from_tensor_slices(dict(article))

ratings = ratings.map(lambda x: {
    "Title": x["Title"],
    "User_id": x["User_id"],
    "Like": float(x["Like"])
})

articles = articles.map(lambda x: x["Title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = ratings.take(35_000)
test = ratings.take(8_188)

print('Total Data: {}'.format(len(ratings)))

article_titles = articles.batch(1_000)
user_ids = ratings.batch(1_000).map(lambda x: x["User_id"])

unique_article_titles = np.unique(np.concatenate(list(article_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print('Unique articles: {}'.format(len(unique_article_titles)))
print('Unique users: {}'.format(len(unique_user_ids)))

class ArticleModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 64

    # User and article models.
    self.article_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_article_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_article_titles) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ]) #130, 100

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.article_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["User_id"])
    # And pick out the movie features and pass them into the movie model.
    article_embeddings = self.article_model(features["Title"])

    return (
        user_embeddings,
        article_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, article_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("Like")

    user_embeddings, article_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, article_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)

model = ArticleModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(100_000).batch(1_000).cache()
cached_test = test.batch(1_000).cache()

#model_checkpoint=tf.keras.callbacks.ModelCheckpoint('CIFAR10{epoch:02d}.h5',period=5,save_weights_only=True)

history = model.fit(cached_train, epochs=1000)

# Plot the loss and accuracy curves for training and validation
plt.plot(history.history['total_loss'], color='b', label="Loss")
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

metrics = model.evaluate(cached_test, return_dict=True)

print(f"\nRetrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}")

def predict_article(user, top_n=3):
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
      tf.data.Dataset.zip((articles.batch(100), articles.batch(100).map(model.article_model)))
    )

    # Get recommendations.
    _, titles = index(tf.constant([str(user)]))

    data = []
    # return data as array
    # print('Top {} recommendations for user {}:\n'.format(top_n, user))
    for i, title in enumerate(titles[0, :top_n].numpy()):
        result = title.decode("utf-8")
        data.append(result)
    
    return data    

# predict_article(132,5)

#Save model
# model.save_weights('tfrs.h5')
