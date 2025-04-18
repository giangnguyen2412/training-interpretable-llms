# Data Attribution

## Definition
Data attribution refers to methods that connect the behavior of a machine learning model—whether that behavior is predictive, generative, or evaluative—to the data used during its training.

## Notions of Data Attribution

### Corroborative Attribution

Corroborative attribution focuses on finding support for a model’s output without necessarily implying causation. It is used, for example, in applications such as:

* Citation Generation: When a large language model generates an assertion, corroborative attribution may surface examples that logically support the claim, much like providing references.

* Copyright Infringement Detection: By retrieving visually or textually similar content from a corpus, one can judge the originality of generated content.

An example of corroborative attribution is OlmoTrace (n-gram overlap retrieval)
Another example is training nearest neighbors to retrieve similar examples from the training set, which can be used to support or refute a model's output.

### Game-Theoretic Attribution

Game-theoretic attribution aims to assign a "fair" value or credit to each datapoint in the training set based on its contribution to the model's performance or utility. This is especially important in two key scenarios:

* Data Pricing: Determining a fair compensation for data providers (e.g., paying artists for using their work in generative models).

* Liability Assignment: Attributing blame or responsibility when a model fails or produces harmful outcomes.

### Predictive Attribution

Predictive attribution, the primary focus of the tutorial, shifts the question to “what would have happened if we changed the training data?” The idea is to predict model behavior given a particular training subset.

Applications include:

* Data Selection: Choosing the optimal dataset to maximize model performance.

* Sensitivity Analysis: Determining the number of samples that need to be removed or modified to change a model’s statistical conclusions.

Predictive attribution uses datamodels—functions that, given a training set, can predict how the model would behave (e.g., through loss or output predictions).