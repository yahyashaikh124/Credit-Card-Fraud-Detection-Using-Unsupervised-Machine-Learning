README



The notebooks are designed to run on the Kaggle Platform.



**Running the Code**



1. Log in to Kaggle and upload the notebooks.
2. Within the Kaggle notebook, go to Add Input → Competitions → IEEE-CIS Fraud Detection to attach the competition dataset.
3. All notebooks can run on any Kaggle runtime except clustering-dbscan.This notebook requires TPU VM v3-8 due to high memory demands (peaking at ~300 GB RAM).



**Notebooks Overview**



**EDA \& Data Preprocessing:** Performs exploratory data analysis and preprocessing.



**supervised-unsupervised:** Uses unsupervised methods (K-Means, Isolation Forest, PCA) to generate features, which are then fed into a supervised LightGBM model—yielding improved results, especially in terms of precision.



**ieee-autoencoder:** Implements an Autoencoder model.



**ieee-methods-experiments.ipynb:** Contains experiments with:



**K-Means clustering**

* Isolation Forest
* Isolation Forest with SHAP
* Deep Autoencoding Gaussian Mixture Model (DAGMM)
* Hypergraph Neural Networks (HGNN)
* Ensemble models (DAGMM + HGNN)
* DCSAN (Contamination-Suppression Autoencoder with latent clustering)
* Multi-Feature Multi-Layer Perceptron (MF-MLP)



**clustering-dbscan:** Experiments with Density-Based Clustering of Spatial Applications with Noise (DBSCAN).



**k-means diagram:** Shows data in 2D K-means space and a diagram with data points and distance from centroid. 



