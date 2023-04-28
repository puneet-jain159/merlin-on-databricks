# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_merlin_01-building-recommender-systems-with-merlin/nvidia_logo.png" style="width: 90px; float: right;"> 
# MAGIC
# MAGIC ## Building Intelligent Recommender Systems with Merlin
# MAGIC
# MAGIC This notebook has been tested on DBR 13 ML 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overview
# MAGIC
# MAGIC Recommender Systems (RecSys) are the engine of the modern internet and the catalyst for human decisions. Building a recommendation system is challenging because it requires multiple stages (data preprocessing, offline training, item retrieval, filtering, ranking, ordering, etc.) to work together seamlessly and efficiently. The biggest challenges for new practitioners are the lack of understanding around what RecSys look like in the real world, and the gap between examples of simple models and a production-ready end-to-end recommender systems.

# COMMAND ----------

# MAGIC %md
# MAGIC The figure below represents a four-stage recommender systems. This is a more complex process than only training a single model and deploying it, and it is much more realistic and closer to what's happening in the real-world recommender production systems.

# COMMAND ----------

# MAGIC %md
# MAGIC ![fourstage](https://raw.githubusercontent.com/NVIDIA-Merlin/Merlin/main/examples/images/fourstages.png)

# COMMAND ----------

# MAGIC %md
# MAGIC In these series of notebooks, we are going to showcase how we can deploy a four-stage recommender systems using Merlin Systems library easily on [Triton Inference Server](https://github.com/triton-inference-server/server). Let's go over the concepts in the figure briefly. 
# MAGIC - **Retrieval:** This is the step to narrow down millions of items into thousands of candidates. We are going to train a Two-Tower item retrieval model to retrieve the relevant top-K candidate items.
# MAGIC - **Filtering:** This step is to exclude the already interacted  or undesirable items from the candidate items set or to apply business logic rules. Although this is an important step, for this example we skip this step.
# MAGIC - **Scoring:** This is also known as ranking. Here the retrieved and filtered candidate items are being scored. We are going to train a ranking model to be able to use at our scoring step. 
# MAGIC - **Ordering:** At this stage, we can order the final set of items that we want to recommend to the user. Here, we’re able to align the output of the model with business needs, constraints, or criteria.
# MAGIC
# MAGIC To learn more about the four-stage recommender systems, you can listen to Even Oldridge's [Moving Beyond Recommender Models talk](https://www.youtube.com/watch?v=5qjiY-kLwFY&list=PL65MqKWg6XcrdN4TJV0K1PdLhF_Uq-b43&index=7) at KDD'21 and read more [in this blog post](https://eugeneyan.com/writing/system-design-for-discovery/).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Learning objectives
# MAGIC - Understanding four stages of recommender systems
# MAGIC - Training retrieval and ranking models with Merlin Models
# MAGIC - Setting up feature store and approximate nearest neighbours (ANN) search libraries
# MAGIC - Deploying trained models to Triton Inference Server with Merlin Systems

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Neccessary CUDA files

# COMMAND ----------

# MAGIC %pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import required libraries and functions

# COMMAND ----------

# MAGIC %md
# MAGIC **Compatibility:**
# MAGIC
# MAGIC This notebook is developed and tested using the latest `merlin-tensorflow` container from the NVIDIA NGC catalog. To find the tag for the most recently-released container, refer to the [Merlin TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow) page.

# COMMAND ----------

# for running this example on GPU, install the following libraries

# for running this example on CPU, uncomment the following lines
# %pip install tensorflow-cpu 
# %pip uninstall cudf


# COMMAND ----------

import os
import nvtabular as nvt
from nvtabular.ops import Rename, Filter, Dropna, LambdaOp, Categorify, \
    TagAsUserFeatures, TagAsUserID, TagAsItemFeatures, TagAsItemID, AddMetadata

from merlin.schema.tags import Tags

import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.datasets.ecommerce import transform_aliccp
import tensorflow as tf

# for running this example on CPU, comment out the line below
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# COMMAND ----------

# disable INFO and DEBUG logging everywhere
import logging

logging.disable(logging.WARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC In this example notebook, we will generate the synthetic train and test datasets mimicking the real [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1) dataset to build our recommender system models.
# MAGIC
# MAGIC First, we define our input path and feature repo path.

# COMMAND ----------

DATA_FOLDER = os.environ.get("DATA_FOLDER", "/workspace/data/")
# set up the base dir for feature store
BASE_DIR = "/dbfs/puneet.jain@databricks.com/Merlin/examples/Building-and-deploying-multi-stage-RecSys/"

# COMMAND ----------

# MAGIC %md
# MAGIC Then, we use `generate_data` utility function to generate synthetic dataset. 

# COMMAND ----------

from merlin.datasets.synthetic import generate_data

NUM_ROWS = os.environ.get("NUM_ROWS", 100_000)
train_raw, valid_raw = generate_data("aliccp-raw", int(NUM_ROWS), set_sizes=(0.7, 0.3))

# COMMAND ----------

# MAGIC %md
# MAGIC If you would like to use the real ALI-CCP dataset, you can use [get_aliccp()](https://github.com/NVIDIA-Merlin/models/blob/main/merlin/datasets/ecommerce/aliccp/dataset.py) function instead. This function takes the raw csv files, and generate parquet files that can be directly fed to NVTabular workflow above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering with NVTabular

# COMMAND ----------

output_path = os.path.join(DATA_FOLDER, "processed_nvt")

# COMMAND ----------

# MAGIC %md
# MAGIC In the following NVTabular workflow, notice that we apply the `Dropna()` Operator at the end. We add the Operator to remove rows with missing values in the final DataFrame after the preceding transformations. Although, the synthetic dataset that we generate and use in this notebook does not have null entries, you might have null entries in your `user_id` and `item_id` columns in your own custom dataset. Therefore, while applying `Dropna()` we will not be registering null `user_id_raw` and `item_id_raw` values in the feature store, and will be avoiding potential issues that can occur because of any null entries.

# COMMAND ----------

user_id_raw = ["user_id"] >> Rename(postfix='_raw') >> LambdaOp(lambda col: col.astype("int32")) >> TagAsUserFeatures()
item_id_raw = ["item_id"] >> Rename(postfix='_raw') >> LambdaOp(lambda col: col.astype("int32")) >> TagAsItemFeatures()

user_id = ["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()
item_id = ["item_id"] >> Categorify(dtype="int32") >> TagAsItemID()

item_features = (
    ["item_category", "item_shop", "item_brand"] >> Categorify(dtype="int32") >> TagAsItemFeatures()
)

user_features = (
    [
        "user_shops",
        "user_profile",
        "user_group",
        "user_gender",
        "user_age",
        "user_consumption_2",
        "user_is_occupied",
        "user_geography",
        "user_intentions",
        "user_brands",
        "user_categories",
    ] >> Categorify(dtype="int32") >> TagAsUserFeatures()
)

targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

outputs = user_id + item_id + item_features + user_features + user_id_raw + item_id_raw + targets

# add dropna op to filter rows with nulls
outputs = outputs >> Dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's call `transform_aliccp` utility function to be able to perform `fit` and `transform` steps on the raw dataset applying the operators defined in the NVTabular workflow pipeline below, and also save our workflow model. After fit and transform, the processed parquet files are saved to output_path.

# COMMAND ----------

transform_aliccp(
    (train_raw, valid_raw), output_path, nvt_workflow=outputs, workflow_name="workflow"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training a Retrieval Model with Two-Tower Model

# COMMAND ----------

# MAGIC %md
# MAGIC We start with the offline candidate retrieval stage. We are going to train a Two-Tower model for item retrieval. To learn more about the Two-tower model you can visit [05-Retrieval-Model.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/05-Retrieval-Model.ipynb).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering with NVTabular

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to process our raw categorical features by encoding them using `Categorify()` operator and tag the features with `user` or `item` tags in the schema file. To learn more about [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and the schema object visit this example [notebook](https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb) in the Merlin Models repo.

# COMMAND ----------

# MAGIC %md
# MAGIC Define a new output path to store the filtered datasets and schema files.

# COMMAND ----------

output_path2 = os.path.join(DATA_FOLDER, "processed/retrieval")

# COMMAND ----------

train_tt = Dataset(os.path.join(output_path, "train", "*.parquet"))
valid_tt = Dataset(os.path.join(output_path, "valid", "*.parquet"))

# COMMAND ----------

# MAGIC %md
# MAGIC We select only positive interaction rows where `click==1` in the dataset with `Filter()` operator.

# COMMAND ----------

inputs = train_tt.schema.column_names
outputs = inputs >> Filter(f=lambda df: df["click"] == 1)

workflow2 = nvt.Workflow(outputs)

workflow2.fit(train_tt)

workflow2.transform(train_tt).to_parquet(
    output_path=os.path.join(output_path2, "train")
)

workflow2.transform(valid_tt).to_parquet(
    output_path=os.path.join(output_path2, "valid")
)

# COMMAND ----------

# MAGIC %md
# MAGIC NVTabular exported the schema file, `schema.pbtxt` a protobuf text file, of our processed dataset. To learn more about the schema object and schema file you can explore [02-Merlin-Models-and-NVTabular-integration.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb) notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC **Read filtered parquet files as Dataset objects.**

# COMMAND ----------

train_tt = Dataset(os.path.join(output_path2, "train", "*.parquet"), part_size="500MB")
valid_tt = Dataset(os.path.join(output_path2, "valid", "*.parquet"), part_size="500MB")

# COMMAND ----------

schema = train_tt.schema.select_by_tag([Tags.ITEM_ID, Tags.USER_ID, Tags.ITEM, Tags.USER]).without(['user_id_raw', 'item_id_raw', 'click'])
train_tt.schema = schema
valid_tt.schema = schema

# COMMAND ----------

model_tt = mm.TwoTowerModel(
    schema,
    query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
    samplers=[mm.InBatchSampler()],
    embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
)

# COMMAND ----------

model_tt.compile(
    optimizer="adam",
    run_eagerly=False,
    loss="categorical_crossentropy",
    metrics=[mm.RecallAt(10), mm.NDCGAt(10)],
)
model_tt.fit(train_tt, validation_data=valid_tt, batch_size=1024 * 8, epochs=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exporting query (user) model

# COMMAND ----------

# MAGIC %md
# MAGIC We export the query tower to use it later during the model deployment stage with Merlin Systems.

# COMMAND ----------

query_tower = model_tt.retrieval_block.query_block()
query_tower.save(os.path.join(BASE_DIR, "query_tower"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training a Ranking Model with DLRM

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will move onto training an offline ranking model. This ranking model will be used for scoring our retrieved items.

# COMMAND ----------

# MAGIC %md
# MAGIC Read processed parquet files. We use the `schema` object to define our model.

# COMMAND ----------

# define train and valid dataset objects
train = Dataset(os.path.join(output_path, "train", "*.parquet"), part_size="500MB")
valid = Dataset(os.path.join(output_path, "valid", "*.parquet"), part_size="500MB")

# define schema object
schema = train.schema.without(['user_id_raw', 'item_id_raw'])

# COMMAND ----------

target_column = schema.select_by_tag(Tags.TARGET).column_names[0]
target_column

# COMMAND ----------

# MAGIC %md
# MAGIC Deep Learning Recommendation Model [(DLRM)](https://arxiv.org/abs/1906.00091) architecture is a popular neural network model originally proposed by Facebook in 2019. The model was introduced as a personalization deep learning model that uses embeddings to process sparse features that represent categorical data and a multilayer perceptron (MLP) to process dense features, then interacts these features explicitly using the statistical techniques proposed in [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074). To learn more about DLRM architetcture please visit `Exploring-different-models` [notebook](https://github.com/NVIDIA-Merlin/models/blob/main/examples/04-Exporting-ranking-models.ipynb) in the Merlin Models GH repo.

# COMMAND ----------

model = mm.DLRMModel(
    schema,
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(target_column),
)

# COMMAND ----------

model.compile(optimizer="adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
model.fit(train, validation_data=valid, batch_size=16 * 1024)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save our DLRM model to be able to load back at the deployment stage. 

# COMMAND ----------

model.save(os.path.join(BASE_DIR, "dlrm"))

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we are going to export the required user and item features files, and save the query (user) tower model and item embeddings to disk. If you want to read more about exporting retrieval models, please visit [05-Retrieval-Model.ipynb](https://github.com/NVIDIA-Merlin/models/blob/main/examples/05-Retrieval-Model.ipynb) notebook in Merlin Models library repo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exporting user and item features

# COMMAND ----------

from merlin.models.utils.dataset import unique_rows_by_features

user_features = (
    unique_rows_by_features(train, Tags.USER, Tags.USER_ID)
    .compute()
    .reset_index(drop=True)
)

# COMMAND ----------

user_features.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We will artificially add `datetime` and `created` timestamp columns to our user_features dataframe. This required by Feast to track the user-item features and their creation time and to determine which version to use when we query Feast.

# COMMAND ----------

from datetime import datetime

user_features["datetime"] = datetime.now()
user_features["datetime"] = user_features["datetime"].astype("datetime64[ns]")
user_features["created"] = datetime.now()
user_features["created"] = user_features["created"].astype("datetime64[ns]")

# COMMAND ----------

user_features.head()

# COMMAND ----------

user_features.to_parquet(
    os.path.join(BASE_DIR, "user_features.parquet")
)

# COMMAND ----------

item_features = (
    unique_rows_by_features(train, Tags.ITEM, Tags.ITEM_ID)
    .compute()
    .reset_index(drop=True)
)

# COMMAND ----------

item_features["datetime"] = datetime.now()
item_features["datetime"] = item_features["datetime"].astype("datetime64[ns]")
item_features["created"] = datetime.now()
item_features["created"] = item_features["created"].astype("datetime64[ns]")

# COMMAND ----------

item_features.head()

# COMMAND ----------

# save to disk
item_features.to_parquet(
    os.path.join(BASE_DIR, "item_features.parquet")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract and save Item embeddings

# COMMAND ----------

item_embs = model_tt.item_embeddings(
    Dataset(item_features, schema=schema), batch_size=1024
)
item_embs_df = item_embs.compute(scheduler="synchronous")

# COMMAND ----------

# select only item_id together with embedding columns
item_embeddings = item_embs_df.drop(
    columns=["item_category", "item_shop", "item_brand"]
)

# COMMAND ----------

item_embeddings.head()

# COMMAND ----------

# save to disk
item_embeddings.to_parquet(os.path.join(BASE_DIR, "item_embeddings.parquet"))

# COMMAND ----------


