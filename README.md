# Project Execution Instructions
## Run the model directly

```
python3 src/model.py -t <training> -m <model> -c <countries>`
```
* training - whether training is needed and mode - test or prod
* model - which model to use RandomForestRegressor or ExtraTreesRegressor
* countries - which countries to predict revenue

## Run the unittests
### Run all tests
```
python3 run-tests.py
```
### Model Tests
All tests - `python3 -m unittest unittests/ModelTests.py`
Specific test - `python3 -m unittest unittests.ModelTest.test_02_load`

### Logger Tests
All tests - `python3 -m unittest unittests/LoggerTests.py`
Specific test - `python3 -m unittest unittests.ModelTest.test_02_predict`

# IBM AI Enterprise Workflow Capstone
Files for the IBM AI Enterprise Workflow Capstone project. 

### Case study part 1 - The notebook capstone-case-study.ipynb contains all the findings

### Case study part 2 - The code provided in solution-guidance has been modified to take model as parameter and also work for specific set of countries. The notebook capstone-case-study.ipynb contains details for iterating on different models. Running with different preprocessors (StandardScaler, OneHotEncoder) and models (RandomForestRegressor and ExtraTreesRegressor) the revenue prediction has gone down for some countries while has increased for others.

### Case study part 3

## Outline

1. Build a draft version of an API with train, predict, and logfile endpoints.
2. Using Docker, bundle your API, model, and unit tests.
3. Using test-driven development iterate on your API in a way that anticipates scale, load, and drift.
4. Create a post-production analysis script that investigates the relationship between model performance and the business metric.
5. Articulate your summarized findings in a final report.


At a higher level you are being asked to:

1. Ready your model for deployment
2. Query your API with new data and test your monitoring tools
3. Compare your results to the gold standard


To **ready your model for deployment** you will be required to prepare you model in a way that the Flask API can both 
train and predict.  There are some differences when you compare this model to most of those we have discussed 
throughout this specialization.  When it comes to training one solution is that the model train script simply uses all
files in a given directory.  This way you could set your model up to be re-trained at regular intervals with little 
overhead.  

Prediction in the case of this model requires a little more thought.  You are not simply passing a query corresponding
to a row in a feature matrix, because this business opportunity requires that the API takes a country name and a date.
There are many ways to accommodate these requirements.  You model may simply save the forecasts for a range of dates,
then the 'predict' function serves to looks up the specified 30 day revenue prediction.  You model could also transform
the target date into an appropriate input vector that is then used as input for a trained model.

You might be tempted to setup the predict function to work only with the latest date, which would be appropriate in 
some circumstances, but in this case we are building a tool to fit the specific needs of individuals.  Some people in
leadership at AAVAIL make projections at the end of the month and others do this on the 15th so the predict function
needs to work for all of the end users.

In the case of this project you can safely assume that there are only a few individuals that will be active users of 
the model so it may not be worth the effort to optimize for speed when it comes to prediction or training.  The important
thing is to arrive at a working solution.

Once all of your tests pass and your model is being served via Docker you will need to **query the API**.  One suggestion
for this part is to use a script to simulate the process.  You may want to start with fresh log files and then for every
new day make a prediction with the consideration that you have not yet seen the rest of the future data.  To may the 
process more realistic you could 're-train' your model say every week or nightly.  At a minimum you should have predictions
for each day when you are finished and you should compare them to the known values.

To monitor performance there are several plots that could be made.  The time-series plot where X are the day intervals
and Y is the 30 day revenue (projected and known) can be particularly useful here.  Because we obtain labels for y the 
performance of your model can be monitored by comparing predicted and known values.
