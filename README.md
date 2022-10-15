# SSF-TransformerEnhancement

This repo consists implementations for "Surface Semantic Features to Prevent Credit Fraud: A Cross-Layer Fusion Feature Perceptron Based on Transformer-Based User Behavior Profiles". 

## Setup

### Create python environment using:
```
pip install -r requirements.txt
```

### Train on baseline:
```
python train_transformer_baseline.py
```

### Evaluate:
```
python evaluate.py
```
You might need to initialize testset manually during evaluation.


### Run on custom data

Coming soon.


## Acknowledgements
This repo is inspired by [Tensorflow Docs-Xception Module](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception) and [Visformer](https://arxiv.org/abs/2104.12533)

Especially thanks to Zhang(Ph.D., Autonomous Driving Research, Ottawa) for his contribution towards feature engineering.

Further code to be released upon acceptance.
