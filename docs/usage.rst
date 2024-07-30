=====
Usage
=====

XiRescore can be used in different ways. First of all, there are different options for data sources and result targets:

* Parquet files
* CSV files
* XiSearch2 databases
* Pandas DataFrames

Secondly there are different ways of calling XiRescore: Either as a Python module in code or via CLI.

To use XiRescore in Python code use the XiRescore class:

.. autoclass:: xirescore.XiRescore::XiRescore
   :no-index:

XiRescore accepts an option dictionary as configuration. The passed options will be merged with the default options,
such that all existing default values or arrays are replaces. A special case are `rescoring.model_params` which replace the default dictionary if provided.
The available options and default values can be found under :ref:`options`.
