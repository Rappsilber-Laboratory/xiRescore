============
Introduction
============

XiRescore is a tool to improve the separation of tarets and decoy for cross-link mass spectrometry matches (CSMs) using provided features and a native score.

------------
How It Works
------------

The basic method works as follows:

* Select samples with decoys and low FDR targets
* Use these samples to train multiple classifiers using k-fold crossvalidation
* Rescore all samples by using the average score of all classifiers
* For the k-fold samples, use only the score of the classifier they were not trained with

To support large datasets that do not fit into memory, xiRescore rescored the input data in batches and write the results to disk before continuing with the next batch.

-----------
How To Cite
-----------

Coming soon...
