<div align="center">
<!-- Title: -->
  <a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">
  </a>
  <h1><a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">Estimating Credit Spreads Via Machine Learning</a></h1>
<!-- Short description: -->
</div>

## Background

This tool was developed as part of an ongoing research project into the use of machine learning algorithms to predict credit spreads on corporate bonds. It implements a Neural Networks models via TensorFlow as well as a Gradient Boosted Regression Trees model via XGBoost. It also allows for model exploration and interpretation of results by impementing the Eli5 and LIME Libraries. Fundamentally, The tool creates the the ability to parse the raw research data from unstructured Excel Files into a SQLite database so that it can be transformed into a functional machine learning data set. This tool also provides some data analysis ability to examine the data quality. 

For more information on the research, refer to the research.docx document file (currently a work in progress) available here:  https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/tree/master/Results%20April7-2023/Documentation

## Getting Started

run python.exe interface.py at the command line to launch the GUI or run interface.py in your Python IDE. 

IMPORTANT: When running for the first time you must do the following two steps in this order:
1. Build the valuation database:  Menu -> Database -> Build Valuation Curve DB
2. Calculate the z-spreads: Menu -> Database -> Calculate Z-Spread Analytics
