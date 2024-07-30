=======================
Handling Large Datasets
=======================

When handling datasets that might exceed the memory of your machine you should use xiRescore's file source/target capabilities.
XiRescore can load and store CSV/Parquet files in batches. The batch sizes can be configured with the option ``rescoring.spectra_batch_size``.
Notice that the samples used for the k-fold creoss-validation are kept in memory at all time to select the right score for them when their according
spectrum batch is getting rescored.
