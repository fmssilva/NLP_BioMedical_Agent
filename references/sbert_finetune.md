Logo

Search docs
Getting Started

Installation
Quickstart
Migration Guide
Sentence Transformer

Usage
Pretrained Models
Training Overview
Why Finetune?
Training Components
Model
Dataset
Loss Function
Training Arguments
Evaluator
Trainer
Multi-Dataset Training
Deprecated Training
Best Base Embedding Models
Comparisons with CrossEncoder Training
Dataset Overview
Loss Overview
Training Examples
Cross Encoder

Usage
Pretrained Models
Training Overview
Loss Overview
Training Examples
Sparse Encoder

Usage
Pretrained Models
Training Overview
Dataset Overview
Loss Overview
Training Examples
Package Reference

Sentence Transformer
Cross Encoder
Sparse Encoder
util
 Training Overview Edit on GitHub
Training Overview
Why Finetune?
Finetuning Sentence Transformer models often heavily improves the performance of the model on your use case, because each task requires a different notion of similarity. For example, given news articles:

“Apple launches the new iPad”

“NVIDIA is gearing up for the next GPU generation”

Then the following use cases, we may have different notions of similarity:

a model for classification of news articles as Economy, Sports, Technology, Politics, etc., should produce similar embeddings for these texts.

a model for semantic textual similarity should produce dissimilar embeddings for these texts, as they have different meanings.

a model for semantic search would not need a notion for similarity between two documents, as it should only compare queries and documents.

Also see Training Examples for numerous training scripts for common real-world applications that you can adopt.

Training Components
Training Sentence Transformer models involves between 4 to 6 components:

Model
Learn how to initialize the model for training.
Dataset
Learn how to prepare the data for training.
Loss Function
Learn how to prepare and choose a loss function.
Training Arguments
Learn which training arguments are useful.
Evaluator
Learn how to evaluate during and after training.
Trainer
Learn how to start the training process.
Model
Sentence Transformer models consist of a sequence of Modules or Custom Modules, allowing for a lot of flexibility. If you want to further finetune a SentenceTransformer model (e.g. it has a modules.json file), then you don’t have to worry about which modules are used:

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
But if instead you want to train from another checkpoint, or from scratch, then these are the most common architectures you can use:


Transformers
Most Sentence Transformer models use the Transformer and Pooling modules. The former loads a pretrained transformer model (e.g. BERT, RoBERTa, DistilBERT, ModernBERT, etc.) and the latter pools the output of the transformer to produce a single vector representation for each input sentence.

Documentation

sentence_transformers.models.Transformer
sentence_transformers.models.Pooling
from sentence_transformers import models, SentenceTransformer

transformer = models.Transformer("google-bert/bert-base-uncased")
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

model = SentenceTransformer(modules=[transformer, pooling])
This is the default option in Sentence Transformers, so it’s easier to use the shortcut:

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google-bert/bert-base-uncased")
Tip

The strongest base models are often “encoder models”, i.e. models that are trained to produce a meaningful token embedding for inputs. You can find strong candidates here:

fill-mask models - trained for token embeddings

sentence similarity models - trained for text embeddings

feature-extraction models - trained for text embeddings

Consider looking for base models that are designed on your language and/or domain of interest. For example, FacebookAI/xlm-roberta-base will work better than google-bert/bert-base-uncased for Turkish.


Static
Dataset
The SentenceTransformerTrainer trains and evaluates using datasets.Dataset (one dataset) or datasets.DatasetDict instances (multiple datasets, see also Multi-dataset training).


Data on 🤗 Hugging Face Hub
If you want to load data from the Hugging Face Datasets, then you should use datasets.load_dataset():

Documentation

Datasets, Loading from the Hugging Face Hub
datasets.load_dataset()
sentence-transformers/all-nli
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev")

print(train_dataset)
"""
Dataset({
    features: ['premise', 'hypothesis', 'label'],
    num_rows: 942069
})
"""
Some datasets (including sentence-transformers/all-nli) require you to provide a “subset” alongside the dataset name. sentence-transformers/all-nli has 4 subsets, each with different data formats: pair, pair-class, pair-score, triplet.

Note

Many Hugging Face datasets that work out of the box with Sentence Transformers have been tagged with sentence-transformers, allowing you to easily find them by browsing to https://huggingface.co/datasets?other=sentence-transformers. We strongly recommend that you browse these datasets to find training datasets that might be useful for your tasks.


Local Data (CSV, JSON, Parquet, Arrow, SQL)

Local Data that requires pre-processing
Dataset Format
It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format). Verifying whether a dataset format works with a loss function involves two steps:

If your loss function requires a Label according to the Loss Overview table, then your dataset must have a column named “label”, “labels”, “score” or “scores”. This column is automatically taken as the label.

All columns not named “label”, “labels”, “score” or “scores” are considered Inputs according to the Loss Overview table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are irrelevant, only the order matters.

For example, given a dataset with columns ["text1", "text2", "label"] where the “label” column has float similarity score between 0 and 1, we can use it with CoSENTLoss, AnglELoss, and CosineSimilarityLoss because it:

has a “label” column as is required for these loss functions.

has 2 non-label columns, exactly the amount required by these loss functions.

Be sure to re-order your dataset columns with Dataset.select_columns if your columns are not ordered correctly. For example, if your dataset has ["good_answer", "bad_answer", "question"] as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the good_answer column will be taken as the anchor, bad_answer as the positive, and question as the negative.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with Dataset.remove_columns as they will be used as inputs otherwise. You can also use Dataset.select_columns to keep only the desired columns.

Loss Function
Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See Dataset Format to learn what datasets are valid for which loss functions. Additionally, the Loss Overview will be your best friend to learn about the options.

Most loss functions can be initialized with just the SentenceTransformer that you’re training, alongside some optional parameters, e.g.:

Documentation

sentence_transformers.losses.CoSENTLoss

Losses API Reference

Loss Overview

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss

# Load a model to train/finetune
model = SentenceTransformer("xlm-roberta-base")

# Initialize the CoSENTLoss
# This loss requires pairs of text and a float similarity score as a label
loss = CoSENTLoss(model)

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
"""
Dataset({
    features: ['sentence1', 'sentence2', 'label'],
    num_rows: 942069
})
"""
Training Arguments
The SentenceTransformerTrainingArguments class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.

Key Training Arguments for improving training performance
learning_rate
lr_scheduler_type
warmup_ratio
num_train_epochs
max_steps
per_device_train_batch_size
per_device_eval_batch_size
auto_find_batch_size
fp16
bf16
load_best_model_at_end
metric_for_best_model
gradient_accumulation_steps
gradient_checkpointing
eval_accumulation_steps
optim
batch_sampler
multi_dataset_batch_sampler
prompts
router_mapping
learning_rate_mapping

Key Training Arguments for observing training performance
eval_strategy
eval_steps
save_strategy
save_steps
save_total_limit
report_to
run_name
log_level
logging_steps
push_to_hub
hub_model_id
hub_strategy
hub_private_repo

Here is an example of how SentenceTransformerTrainingArguments can be initialized:

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)
Evaluator
You can provide the SentenceTransformerTrainer with an eval_dataset to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model’s performance with useful metrics before, during, or after training. You can use both an eval_dataset and an evaluator, one or the other, or neither. They evaluate based on the eval_strategy and eval_steps Training Arguments.

Here are the implemented Evaluators that come with Sentence Transformers:

Evaluator

Required Data

BinaryClassificationEvaluator

Pairs with class labels.

EmbeddingSimilarityEvaluator

Pairs with similarity scores.

InformationRetrievalEvaluator

Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]).

NanoBEIREvaluator

No data required.

MSEEvaluator

Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts.

ParaphraseMiningEvaluator

Mapping of IDs to sentences & pairs with IDs of duplicate sentences.

RerankingEvaluator

List of {'query': '...', 'positive': [...], 'negative': [...]} dictionaries.

TranslationEvaluator

Pairs of sentences in two separate languages.

TripletEvaluator

(anchor, positive, negative) pairs.

Additionally, SequentialEvaluator should be used to combine multiple evaluators into one Evaluator that can be passed to the SentenceTransformerTrainer.

Sometimes you don’t have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.


EmbeddingSimilarityEvaluator with STSb
Documentation

sentence-transformers/stsb
sentence_transformers.evaluation.EmbeddingSimilarityEvaluator
sentence_transformers.SimilarityFunction
from datasets import load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
# You can run evaluation like so:
# results = dev_evaluator(model)

TripletEvaluator with AllNLI

NanoBEIREvaluator
Tip

When evaluating frequently during training with a small eval_steps, consider using a tiny eval_dataset to minimize evaluation overhead. If you’re concerned about the evaluation set size, a 90-1-9 train-eval-test split can provide a balance, reserving a reasonably sized test set for final evaluations. After training, you can assess your model’s performance using trainer.evaluate(test_dataset) for test loss or initialize a testing evaluator with test_evaluator(model) for detailed test metrics.

If you evaluate after training, but before saving the model, your automatically generated model card will still include the test results.

Warning

When using Distributed Training, the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices.

Trainer
The SentenceTransformerTrainer is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let’s have a look at a script where all of these components come together:

Documentation

SentenceTransformer

SentenceTransformerModelCardData

load_dataset()

MultipleNegativesRankingLoss

SentenceTransformerTrainingArguments

TripletEvaluator

SentenceTransformerTrainer

SentenceTransformer.save_pretrained

SentenceTransformer.push_to_hub

Training Examples

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "microsoft/mpnet-base",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on AllNLI triplets",
    )
)

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"].select(range(100_000))
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="all-nli-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-test",
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub("mpnet-base-all-nli-triplet")
Callbacks
This Sentence Transformers trainer integrates support for various transformers.TrainerCallback subclasses, such as:

WandbCallback to automatically log training metrics to W&B if wandb is installed

TensorBoardCallback to log training metrics to TensorBoard if tensorboard is accessible.

CodeCarbonCallback to track the carbon emissions of your model during training if codecarbon is installed.

Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers Callbacks documentation for more information on the integrated callbacks and how to write your own callbacks.

Multi-Dataset Training
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, sentence_transformers.trainer.SentenceTransformerTrainer can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

Use a dictionary of Dataset instances (or a DatasetDict) as the train_dataset (and optionally also eval_dataset).

(Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the MultiDatasetBatchSamplers enum, which can be passed to the SentenceTransformerTrainingArguments via multi_dataset_batch_sampler. Valid options are:

MultiDatasetBatchSamplers.ROUND_ROBIN: Round-robin sampling from each dataset until one is exhausted. With this strategy, it’s likely that not all samples from each dataset are used, but each dataset is sampled from equally.

MultiDatasetBatchSamplers.PROPORTIONAL (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

This multi-task training has been shown to be very effective, e.g. Huang et al. employed MultipleNegativesRankingLoss, CoSENTLoss, and a variation on MultipleNegativesRankingLoss without in-batch negatives and only hard negatives to reach state-of-the-art performance on Chinese. They even applied MatryoshkaLoss to allow the model to produce Matryoshka Embeddings.

Training on multiple datasets looks like this:

Documentation

datasets.load_dataset()

SentenceTransformer

SentenceTransformerTrainer

CoSENTLoss

MultipleNegativesRankingLoss

SoftmaxLoss

sentence-transformers/all-nli

sentence-transformers/stsb

sentence-transformers/quora-duplicates

sentence-transformers/natural-questions

Training Examples:

Quora Duplicate Questions > Multi-task learning

AllNLI + STSb > Multi-task learning

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss

# 1. Load a model to finetune
model = SentenceTransformer("bert-base-uncased")

# 2. Load several Datasets to train with
# (anchor, positive)
all_nli_pair_train = load_dataset("sentence-transformers/all-nli", "pair", split="train[:10000]")
# (premise, hypothesis) + label
all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train[:10000]")
# (sentence1, sentence2) + score
all_nli_pair_score_train = load_dataset("sentence-transformers/all-nli", "pair-score", split="train[:10000]")
# (anchor, positive, negative)
all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
# (sentence1, sentence2) + score
stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:10000]")
# (anchor, positive)
quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:10000]")
# (query, answer)
natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:10000]")

# We can combine all datasets into a dictionary with dataset names to datasets
train_dataset = {
    "all-nli-pair": all_nli_pair_train,
    "all-nli-pair-class": all_nli_pair_class_train,
    "all-nli-pair-score": all_nli_pair_score_train,
    "all-nli-triplet": all_nli_triplet_train,
    "stsb": stsb_pair_score_train,
    "quora": quora_pair_train,
    "natural-questions": natural_questions_train,
}

# 3. Load several Datasets to evaluate with
# (anchor, positive, negative)
all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
# (sentence1, sentence2, score)
stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
# (anchor, positive)
quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[10000:11000]")
# (query, answer)
natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

# We can use a dictionary for the evaluation dataset too, but we don't have to. We could also just use
# no evaluation dataset, or one dataset.
eval_dataset = {
    "all-nli-triplet": all_nli_triplet_dev,
    "stsb": stsb_pair_score_dev,
    "quora": quora_pair_dev,
    "natural-questions": natural_questions_dev,
}

# 4. Load several loss functions to train with
# (anchor, positive), (anchor, positive, negative)
mnrl_loss = MultipleNegativesRankingLoss(model)
# (sentence_A, sentence_B) + class
softmax_loss = SoftmaxLoss(model, model.get_sentence_embedding_dimension(), 3)
# (sentence_A, sentence_B) + score
cosent_loss = CoSENTLoss(model)

# Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where.
# Note that you can also just use one loss if all of your training/evaluation datasets use the same loss
losses = {
    "all-nli-pair": mnrl_loss,
    "all-nli-pair-class": softmax_loss,
    "all-nli-pair-score": cosent_loss,
    "all-nli-triplet": mnrl_loss,
    "stsb": cosent_loss,
    "quora": mnrl_loss,
    "natural-questions": mnrl_loss,
}

# 5. Define a simple trainer, although it's recommended to use one with args & evaluators
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses,
)
trainer.train()

# 6. save the trained model and optionally push it to the Hugging Face Hub
model.save_pretrained("bert-base-all-nli-stsb-quora-nq")
model.push_to_hub("bert-base-all-nli-stsb-quora-nq")
Deprecated Training
Prior to the Sentence Transformers v3.0 release, models would be trained with the SentenceTransformer.fit() method and a DataLoader of InputExample, which looked something like this:

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer("distilbert/distilbert-base-uncased")

# Define your train examples. You need more than just two examples...
train_examples = [
    InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
    InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
]

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
Since the v3.0 release, using SentenceTransformer.fit() is still possible, but it will initialize a SentenceTransformerTrainer behind the scenes. It is recommended to use the Trainer directly, as you will have more control via the SentenceTransformerTrainingArguments, but existing training scripts relying on SentenceTransformer.fit() should still work.

In case there are issues with the updated SentenceTransformer.fit(), you can also get exactly the old behaviour by calling SentenceTransformer.old_fit() instead, but this method is planned to be deprecated fully in the future.

Best Base Embedding Models
The quality of your text embedding model depends on which transformer model you choose. Sadly we cannot infer from a better performance on e.g. the GLUE or SuperGLUE benchmark that this model will also yield better representations.

To test the suitability of transformer models, I use the training_nli_v2.py script and train on 560k (anchor, positive, negative)-triplets for 1 epoch with batch size 64. I then evaluate on 14 diverse text similarity tasks (clustering, semantic search, duplicate detection etc.) from various domains.

In the following table you find the performance for different models and their performance on this benchmark:

Model	Performance (14 sentence similarity tasks)
microsoft/mpnet-base	60.99
nghuyong/ernie-2.0-en	60.73
microsoft/deberta-base	60.21
roberta-base	59.63
t5-base	59.21
bert-base-uncased	59.17
distilbert-base-uncased	59.03
nreimers/TinyBERT_L-6_H-768_v2	58.27
google/t5-v1_1-base	57.63
nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large	57.31
albert-base-v2	57.14
microsoft/MiniLM-L12-H384-uncased	56.79
microsoft/deberta-v3-base	54.46
Comparisons with CrossEncoder Training
Training SentenceTransformer models is very similar as training CrossEncoder models, with some key differences:

For CrossEncoder training, you can use (variably sized) lists of texts in a column. In SentenceTransformer training, you cannot use lists of inputs (e.g. texts) in a column of the training/evaluation dataset(s). In short: training with a variable number of negatives is not supported.

See the Cross Encoder > Training Overview documentation for more details on training CrossEncoder models.

© Copyright 2026.

Built with Sphinx using a theme provided by Read the Docs.