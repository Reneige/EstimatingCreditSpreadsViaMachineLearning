<div align="center">
<!-- Title: -->
  <a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">
  </a>
  <h1><a href="https://github.com/Reneige/EstimatingCreditSpreadsViaMachineLearning/">Estimating Credit Spreads Via Machine Learning</a></h1>
<!-- Short description: -->
</div>

## Background

This tool was developed as part of an ongoing research project into the use of machine learning algorithms to predict credit spreads on corporate bonds. The current version of this tool implements GUI driven functionality to parse the raw research data from Excel files in the GBP Bonds folder and inject it into an SQLite database. The tool also offers some GUI driven data exploration tools. Later versions will implement model training and model prediction features. 

## Getting Started

run python.exe interface.py at the command line to launch the GUI or run interface.py in your Python IDE. 

IMPORTANT: When running for the first time you must do the following two steps in this order:
1. Build the valuation database:  Menu -> Database -> Build Valuation Curve DB
2. Calculate the z-spreads: Menu -> Database -> Calculate Z-Spread Analytics
