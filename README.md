# Credit-Card-Fraud-Detection-Using-Unsupervised-Machine-Learning
Major Research Project for Masters in Data Science

**Background**

Credit Card Fraudulent Transactions are < 1% of all transactions. In 2022 they accounted for $ 34.4b of global losses. This imbalance makes traditional supervised models biased toward “non-fraud” , especially when patterns change. Unsupervised methods can help flag unusual behavior without labels. This study uses the IEEE-CIS dataset to compare unsupervised detectors and  hybrid (unsupervised + supervised) approaches that feeds their signals into supervised models, aiming to catch more fraud with fewer false alarms.

**Methodology**

The preprocessing profiles missingness, adds null flags, reduces redundancy among highly correlated V-features, and frequency-encodes high-cardinality categorical columns.

<img width="301" height="163" alt="image" src="https://github.com/user-attachments/assets/7238e941-0bf3-4641-a674-3721464071b9" />

We first build unsupervised baselines—K-Means, Isolation Forest, and Autoencoder—using distance-to-centroid, anomaly, and reconstruction-error scores. We then expand the unsupervised set with DBSCAN (Density-Based Clustering) and other methods including Neural Networks and Graphs.

<img width="222" height="162" alt="image" src="https://github.com/user-attachments/assets/14302e30-b7d8-4bc5-8049-087f1493d7cd" />

Finally, we evaluate a hybrid approach: engineer unsupervised signals (K-Means, Isolation Forest, PCA) and form expanded feature set. A supervised LightGBM model is trained on these features. Evaluation using a held-out chronological validation split for Recall and Precision Evaluation, and test set predictions are submitted to Kaggle for Kaggle (Public/Private) 

**Results**

**Unsupervised models.** Isolation Forest led the basic set but with low precision (ROC-AUC 0.747; PR-AUC 0.085; Kaggle Public 0.787/Private 0.726). K-Means and Autoencoder were similar but lower on PR-AUC (0.067, 0.069). In the extended set, DBSCAN achieved the best imbalance-sensitive score for Precision (PR-AUC 0.092). Overall, unsupervised detectors showed limited precision, implying high false-positive rates.

**Hybrid model**(PCA, K-Means,Isolation Forest) +LightGBM produced a large lift: Recall 0.915, Precision 0.529, Kaggle Public 0.927/Private 0.887. False positives dropped by ~50% vs. prior experiments; PCA_0 emerged as the most important feature (≈3× the next). 

<img width="360" height="191" alt="image" src="https://github.com/user-attachments/assets/c0e76172-2a0f-4e7e-a18e-882badec20e6" />


**Conclusion**

Hybrid methods clearly outperformed purely unsupervised anomaly detectors. The hybrid strategy materially increased precision, reducing false positives to levels suitable for real-world deployment. Enriching features with anomaly scores, cluster assignments distances, and PCA components—and then training a supervised learner (LightGBM)—delivered meaningful gains. Beyond precision and recall, evaluations should also consider computational efficiency and resource use so that accuracy is balanced with scalability. E.g., in our setting DBSCAN required ~300 GB RAM and ~30 minutes for training. 

**Running the Code**

Log in to Kaggle and upload the notebooks.

Within the Kaggle notebook, go to Add Input → Competitions → IEEE-CIS Fraud Detection to attach the competition dataset.

All notebooks can run on any Kaggle runtime except clustering-dbscan.This notebook requires TPU VM v3-8 due to high memory demands (peaking at ~300 GB RAM).

**Notebooks Overview**

**EDA & Data Preprocessing**: Performs exploratory data analysis and preprocessing.

**supervised-unsupervised**: Uses unsupervised methods (K-Means, Isolation Forest, PCA) to generate features, which are then fed into a supervised LightGBM model—yielding improved results, especially in terms of precision.

**ieee-autoencoder**: Implements an Autoencoder model.

**ieee-methods-experiments.ipynb**: Contains experiments with:

**K-Means clustering**

Isolation Forest
Isolation Forest with SHAP
Deep Autoencoding Gaussian Mixture Model (DAGMM)
Hypergraph Neural Networks (HGNN)
Ensemble models (DAGMM + HGNN)
DCSAN (Contamination-Suppression Autoencoder with latent clustering)
Multi-Feature Multi-Layer Perceptron (MF-MLP)
clustering-dbscan: Experiments with Density-Based Clustering of Spatial Applications with Noise (DBSCAN).

**k-means diagram:** Shows data in 2D K-means space and a diagram with data points and distance from centroid.
