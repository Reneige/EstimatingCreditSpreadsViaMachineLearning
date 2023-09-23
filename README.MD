<div align="center">
<!-- Title: -->
  <a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">
  </a>
  <h1><a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">Estimating Credit Spreads Via Machine Learning</a></h1>
<!-- Short description: -->
</div>

## Background

This tool was developed to support a research project investigating the application of machine learning algorithms to predict credit spreads in the corporate bond market. It implements a Neural Networks models via TensorFlow as well as a Gradient Boosted Regression Trees model via XGBoost. It also allows for model exploration and prediction interpretation by implementing the Eli5 and LIME Libraries. The tool creates the ability to parse the raw research data from unstructured Excel Files into a SQLite database so that it can be transformed into a functional machine learning data set. This tool also provides some data analysis ability to ensure data quality was at a sufficient level. 

For more information on the research, refer to the research document file available here:  <a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/blob/master/Documentation/Rene%20Alby%20-%20Estimating%20Credit%20Risk%20Premiums%20with%20Gradient%20Boosted%20Regression%20Trees%20and%20Neural%20Networks.pdf"> Estimating Credit Risk Premiums via Gradient Boosted Regression Trees and Neural Networks</a>

## Getting Started

run python.exe interface.py at the command line to launch the GUI or run interface.py in your Python IDE. 

IMPORTANT: When running for the first time you must build the valuation curve database and calculate the z-spreads. This can be performed by running the following steps from the GUI Menu:

1. Build the valuation curve database:  Menu -> Database -> Build Valuation Curve DB
2. Calculate the z-spreads: Menu -> Database -> Calculate Z-Spread Analytics
(must be performed in order specified)

## Screenshot

<img src="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/blob/master/Documentation/AppScreenshot.jpg" alt="Screenshot of Python based Research Application - Rene Alby, May 2023">



Author: Rene Alby
