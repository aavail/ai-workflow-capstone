# Project Execution Instructions
## Run the model directly
`python3 model.py`

## Run the unittests
### Model Tests
All tests - `python3 -m unittest unittests/ModelTests.py`
Specific test - `python3 -m unittest unittests.ModelTest.test_02_load`

### Logger Tests
All tests - `python3 -m unittest unittests/LoggerTests.py`
Specific test - `python3 -m unittest unittests.ModelTest.test_02_predict`

# IBM AI Enterprise Workflow Capstone
Files for the IBM AI Enterprise Workflow Capstone project. 

### Case study part 1 - The notebook capstone-case-study.ipynb contains all the findings

### Case study part 2 - The code provided in solution-guidance has been modified to take model as parameter and also work for specific set of countries. The notebook capstone-case-study.ipynb contains details for iterating on different models.

Time-series analysis is a subject area that has many varied methods and a great potential for customized solutions.
We cannot cover the breadth and depth of this important area of data science in a single case study. We do 
however want to use this as a learning opportunity if time-series analysis is new to you.  For those of you who are seasoned 
practitioners in this area, it may be a useful time to hone your skills or try out a more advanced technique like 
Gaussian processes.  The reference materials for more advanced approaches to time-series analysis will occur in their own section
below. If this is your first encounter with time-series data we suggest that that you begin with the supervised learning
approach before trying out the other possible methods. 

## Deliverable goals

1. State the different modeling approaches that you will compare to address the business opportunity.
2. Iterate on your suite of possible models by modifying data transformations, pipeline architectures, hyperparameters 
and other relevant factors.
3. Re-train your model on all of the data using the selected approach and prepare it for deployment.
4. Articulate your findings in a summary report.

## On time-series analysis

We have used TensorFlow, scikit-learn, and Spark ML as the main ways to implement models.  Time-series analysis 
has been around a long time and there are a number of specialized packages and software to help facilitate model 
implementation.  In the case of our business opportunity, it is required that we 
*predict the next point* or determine a reasonable value for next month's revenue.  If we only had revenue, we could 
engineer features with revenue for the previous day, previous week, previous month and previous three months, for example.
This provides features that machine learning models such as random forests or boosting could use to 
capture the underlying patterns or trends in the the data. You will likely spend some time optimizing this feature
engineering task on a case-by-case basis. 

Predicting the next element in a time-series is in line with the other machine learning tasks that we have encountered in
this specialization.  One caveat to this approach is that sometimes we wish to project further into the future. Although,
it is not a specific request of management in the case of this business opportunity, you may want to consider forecasting 
multiple points into the future, say three months or more. To do this, you have two main categories of methods: 'recursive forecasting' and 'ensemble forecasting'.

In recursive forecasting, you will append your predictions to the feature matrix and *roll* forward until you get to the 
desired number of forecasts in the future.  In the ensemble approach, you will use separate models for each point.  It 
is possible to use a hybridization of these two ideas as well.  If you wish to take your forecasting model to the next
level, try to project several months into the future with one or both of these ideas.

Also, be aware that the assumptions of line regression are generally invalidated when using time-series data because of auto-correlation.  The engineered features are derived mostly from revenue which often means that there is a high degree of correlation.  You will get further with more sophisticated models to in combination with smartly engineered features. 


## Commonly used time-series tools

  * [statsmodels time-series package](https://www.statsmodels.org/dev/tsa.html) - one of the most commonly used 
  time-series analysis packages in Python.  There are a suite of models including autoregressive models (AR), 
  vector autoregressive models (VAR), univariate autoregressive moving average models (ARMA) and more.
  * [Tensorflow time series tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
  * [Prophet](https://research.fb.com/prophet-forecasting-at-scale/)
  
## More advanced methods for time-series analysis

  * [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
  * [Bayesian Methods for time-series](https://docs.pymc.io/api/distributions/timeseries.html)
  * [Gaussian process regression](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html)

## Working with time-series data

  * [scikit-learn MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
  * [NumPy datetime](https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html)
  * [Pandas time-series](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
  * [matplotlib time-series plot](https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/date.html)
  * [scikit-learn time-series train-test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

## Additional materials

  * [Intro paper to Gaussian Processes in time-series](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2011.0550)
  * [Paper for using wavelets to aid time-series forecasts](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142064)
  
## Part 3

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
