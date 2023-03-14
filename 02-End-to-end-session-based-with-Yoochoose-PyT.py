# Databricks notebook source
# Copyright 2022 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Each user is responsible for checking the content of datasets and the
# applicable licenses and determining if suitable for the intended use.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://developer.download.nvidia.com/notebooks/dlsw-notebooks/merlin_transformers4rec_end-to-end-session-based-02-end-to-end-session-based-with-yoochoose-pyt/nvidia_logo.png" style="width: 90px; float: right;">
# MAGIC 
# MAGIC # End-to-end session-based recommendations with PyTorch

# COMMAND ----------

# MAGIC %md
# MAGIC **Start a GPU CLuster and run the below magic commmand**
# MAGIC ```
# MAGIC %pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
# MAGIC %pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com

# COMMAND ----------

# MAGIC %pip install merlin-models nvtabular transformers4rec[pytorch,nvtabular,dataloader]

# COMMAND ----------

# MAGIC %md
# MAGIC In recent years, several deep learning-based algorithms have been proposed for recommendation systems while its adoption in industry deployments have been steeply growing. In particular, NLP inspired approaches have been successfully adapted for sequential and session-based recommendation problems, which are important for many domains like e-commerce, news and streaming media. Session-Based Recommender Systems (SBRS) have been proposed to model the sequence of interactions within the current user session, where a session is a short sequence of user interactions typically bounded by user inactivity. They have recently gained popularity due to their ability to capture short-term or contextual user preferences towards items. 
# MAGIC 
# MAGIC The field of NLP has evolved significantly within the last decade, particularly due to the increased usage of deep learning. As a result, state of the art NLP approaches have inspired RecSys practitioners and researchers to adapt those architectures, especially for sequential and session-based recommendation problems. Here, we leverage one of the state-of-the-art Transformer-based architecture, [XLNet](https://arxiv.org/abs/1906.08237) with Masked Language Modeling (MLM) training technique (see our [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) for details) for training a session-based model.
# MAGIC 
# MAGIC In this end-to-end-session-based recommnender model example, we use `Transformers4Rec` library, which leverages the popular [HuggingFace’s Transformers](https://github.com/huggingface/transformers) NLP library and make it possible to experiment with cutting-edge implementation of such architectures for sequential and session-based recommendation problems. For detailed explanations of the building blocks of Transformers4Rec meta-architecture visit [getting-started-session-based](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/getting-started-session-based) and [tutorial](https://github.com/NVIDIA-Merlin/Transformers4Rec/tree/main/examples/tutorial) example notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model definition using Transformers4Rec

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous notebook, we have created sequential features and saved our processed data frames as parquet files. Now we use these processed parquet files to train a session-based recommendation model with the XLNet architecture.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Get the schema 

# COMMAND ----------

# MAGIC %md
# MAGIC The library uses a schema format to configure the input features and automatically creates the necessary layers. This *protobuf* text file contains the description of each input feature by defining: the name, the type, the number of elements of a list column,  the cardinality of a categorical feature and the min and max values of each feature. In addition, the annotation field contains the tags such as specifying the `continuous` and `categorical` features, the `target` column or the `item_id` feature, among others.

# COMMAND ----------

from utils.merlin_utils import Schema
SCHEMA_PATH = "schema_demo.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)
!cat $SCHEMA_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC We can select the subset of features we want to use for training the model by their tags or their names.

# COMMAND ----------

schema = schema.select_by_name(
   ['item_id-list_seq', 'category-list_seq', 'product_recency_days_log_norm-list_seq', 'et_dayofweek_sin-list_seq']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Define the end-to-end Session-based Transformer-based recommendation model

# COMMAND ----------

# MAGIC %md
# MAGIC For defining a session-based recommendation model, the end-to-end model definition requires four steps:
# MAGIC 
# MAGIC 1. Instantiate [TabularSequenceFeatures](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.features.html?highlight=tabularsequence#transformers4rec.tf.features.sequence.TabularSequenceFeatures) input-module from schema to prepare the embedding tables of categorical variables and project continuous features, if specified. In addition, the module provides different aggregation methods (e.g. 'concat', 'elementwise-sum') to merge input features and generate the sequence of interactions embeddings. The module also supports language modeling tasks to prepare masked labels for training and evaluation (e.g: 'mlm' for masked language modeling) 
# MAGIC 
# MAGIC 2. Next, we need to define one or multiple prediction tasks. For this demo, we are going to use [NextItemPredictionTask](https://nvidia-merlin.github.io/Transformers4Rec/main/api/transformers4rec.tf.model.html?highlight=nextitem#transformers4rec.tf.model.prediction_task.NextItemPredictionTask) with `Masked Language modeling`: during training, randomly selected items are masked and predicted using the unmasked sequence items. For inference, it is meant to always predict the next item to be interacted with.
# MAGIC 
# MAGIC 3. Then we construct a `transformer_config` based on the architectures provided by [Hugging Face Transformers](https://github.com/huggingface/transformers) framework. </a>
# MAGIC 
# MAGIC 4. Finally we link the transformer-body to the inputs and the prediction tasks to get the final pytorch `Model` class.
# MAGIC     
# MAGIC For more details about the features supported by each sub-module, please check out the library [documentation](https://nvidia-merlin.github.io/Transformers4Rec/main/index.html) page.

# COMMAND ----------

from transformers4rec import torch as tr

max_sequence_length, d_model = 20, 320
# Define input module to process tabular input-features and to prepare masked inputs
input_module = tr.TabularSequenceFeatures.from_schema(
    schema,
    max_sequence_length=max_sequence_length,
    continuous_projection=64,
    aggregation="concat",
    d_output=d_model,
    masking="mlm",
)

# Define Next item prediction-task 
prediction_task = tr.NextItemPredictionTask(weight_tying=True)

# Define the config of the XLNet Transformer architecture
transformer_config = tr.XLNetConfig.build(
    d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
)

# Get the end-to-end model 
model = transformer_config.to_torch_model(input_module, prediction_task)

# COMMAND ----------

# MAGIC %md
# MAGIC You can print out the model structure by uncommenting the line below.

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3. Daily Fine-Tuning: Training over a time window¶

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the model is defined, we are going to launch training. For that, Transfromers4rec extends HF Transformers Trainer class to adapt the evaluation loop for session-based recommendation task and the calculation of ranking metrics. The original `train()` method is not modified meaning that we leverage the efficient training implementation from that library, which manages, for example, half-precision (FP16) training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set the training arguments

# COMMAND ----------

# MAGIC %md
# MAGIC An additional argument `data_loader_engine` is defined to automatically load the features needed for training using the schema. The default value is `merlin` for optimized GPU-based data-loading.  Optionally a `PyarrowDataLoader` (`pyarrow`) can also be used as a basic option, but it is slower and works only for small datasets, as the full data is loaded to CPU memory.

# COMMAND ----------

training_args = tr.trainer.T4RecTrainingArguments(
            output_dir="./tmp",
            max_sequence_length=20,
            data_loader_engine='merlin',
            num_train_epochs=10, 
            dataloader_drop_last=False,
            per_device_train_batch_size = 384,
            per_device_eval_batch_size = 512,
            learning_rate=0.0005,
            fp16=True,
            report_to = [],
            logging_steps=200
        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Instantiate the trainer

# COMMAND ----------

recsys_trainer = tr.Trainer(
    model=model,
    args=training_args,
    schema=schema,
    compute_metrics=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Launch daily training and evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC In this demo, we will use the `fit_and_evaluate` method that allows us to conduct a time-based finetuning by iteratively training and evaluating using a sliding time window: At each iteration, we use the training data of a specific time index $t$ to train the model; then we evaluate on the validation data of the next index $t + 1$. Particularly, we set start time to 178 and end time to 180.

# COMMAND ----------

from transformers4rec.torch.utils.examples_utils import fit_and_evaluate
import os
OT_results = fit_and_evaluate(recsys_trainer, start_time_index=178, end_time_index=180, input_dir=os.path.join("/tmp","output/preproc_sessions_by_day"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize the average of metrics over time

# COMMAND ----------

# MAGIC %md
# MAGIC `OT_results` is a list of scores (accuracy metrics) for evaluation based on given start and end time_index. Since in this example we do evaluation on days 179, 180 and 181, we get three metrics in the list one for each day.

# COMMAND ----------

OT_results

# COMMAND ----------

import numpy as np
# take the average of metric values over time
avg_results = {k: np.mean(v) for k,v in OT_results.items()}
for key in sorted(avg_results.keys()): 
    print(" %s = %s" % (key, str(avg_results[key]))) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save the model

# COMMAND ----------

recsys_trainer._save_model_and_checkpoint(save_model_class=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## References

# COMMAND ----------

# MAGIC %md
# MAGIC - Merlin Transformers4rec: https://github.com/NVIDIA-Merlin/Transformers4Rec
# MAGIC 
# MAGIC - Merlin NVTabular: https://github.com/NVIDIA-Merlin/NVTabular/tree/main/nvtabular
# MAGIC 
# MAGIC - Merlin Dataloader: https://github.com/NVIDIA-Merlin/dataloader
# MAGIC 
# MAGIC - Triton inference server: https://github.com/triton-inference-server
