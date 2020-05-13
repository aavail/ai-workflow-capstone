# Solution for *AI in Production* (coursera.org)

## 1. Running the application
To start this application run the following command:
```
python app.py
```
and navigate to the following url: [http://localhost:8080](http://localhost:8080)

**NOTE:** it might take a minute to respond the first time

## 2. Running tests
**NOTE:** Before running the unit tests, make sure the previous command is running

To run all the tests (summary style):
```
python run-tests.py
```
To run all the test (verbose style):
```
python run-tests.py -v
```
To run only the api tests
```
python unittests/ApiTests.py
```
To run only the model tests
```
python unittests/ModelTests.py
```

## 3. (Re)Training the model
A script is available to automate the ingestion of observations (and re-train all models):
```
python solution_guidance/model.py
```
it takes [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) by default, however [Extra Trees Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) is also available as an option when adding the following argument:
```
python solution_guidance/model.py extratrees
```

## 4. Visualizations

As part of the EDA investigation, these graphs were created:

![alt text](static/img/img01.png)
![alt text](static/img/img02.png)
![alt text](static/img/img03.png)
![alt text](static/img/img04.png)
![alt text](static/img/img05.png)

## 5. References
Course link: [learn/ibm-ai-workflow-ai-production](https://www.coursera.org/learn/ibm-ai-workflow-ai-production)

---
The following questions are being evaluated as part of the peer review submission:

1. Are there unit tests for the API?
1. Are there unit tests for the model?
1. Are there unit tests for the logging?
1. Can all of the unit tests be run with a single script and do all of the unit 1. tests pass?
1. Is there a mechanism to monitor performance?
1. Was there an attempt to isolate the read/write unit tests from production 1. models and logs?
1. Does the API work as expected? For example, can you get predictions for a 1. specific country as well as for all countries combined?
1. Does the data ingestion exists as a function or script to facilitate 1. automation?
1. Were multiple models compared?
1. Did the EDA investigation use visualizations?
1. Is everything containerized within a working Docker image?
1. Did they use a visualization to compare their model to the baseline model?
