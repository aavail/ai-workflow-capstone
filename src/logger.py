import time,os,re,csv,sys,uuid,joblib
from datetime import date
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

## model specific variables (iterate the version and note with each change)
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "example random forest on toy data"
SAVED_MODEL = "model-{}.joblib".format(re.sub("\.","_",str(MODEL_VERSION)))
LOG_PATH = os.path.join(os.path.dirname(__file__),'..','log')

def fetch_data():
    """
    example function to fetch data for training
    """
    
    ## import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    y = iris.target

    return(X,y)


def update_predict_log(country,y_pred,y_proba,target_date,runtime,MODEL_VERSION):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    log_file = f"example-predict-{str(today.year)}-{str(today.month)}.log"  
    log_file = os.path.join(LOG_PATH,log_file)
    ## write the data to a csv file    
    header = ['unique_id','timestamp','y_pred','y_proba','target_date','runtime','MODEL_VERSION']
    write_header = False
    if not os.path.exists(log_file):
        write_header = True
    with open(log_file,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),country,y_pred,y_proba,target_date,runtime,MODEL_VERSION])
        writer.writerow(to_write)

       
def update_train_log(country,x_shape,date_range,metric,runtime,MODEL_VERSION, MODEL_VERSION_NOTE) : 
    today = date.today()
    print('-'*50)
    log_file = f"example-training-{str(today.year)}-{str(today.month)}.log"  
    log_file = os.path.join(LOG_PATH,log_file)
    header = ['unique_id','time_stamp','country','trainingset_shape','date_range,metric','runtime,MODEL_VERSION', 'MODEL_VERSION_NOTE']
    with open(log_file,'a') as csvfile:
        writer = csv.writer(csvfile)
    write_header = False
    if not os.path.exists(log_file):
        write_header = True
    with open(log_file,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)
        to_write = map(str,[uuid.uuid4(),time.time(),x_shape,country,date_range,metric,runtime,MODEL_VERSION, MODEL_VERSION_NOTE])
        writer.writerow(to_write)


#def model_train(mode=None):
#    """
#    example funtion to train model
#    
#    'mode' -  can be used to subset data essentially simulating a train
#    """
#
#    ## data ingestion
#    X,y = fetch_data()
#    time_start = time.time()
#    ## Perform a train-test split
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	
#    ## Specify parameters and model
#    params = {'C':1.0,'kernel':'linear','gamma':0.5}
#    clf = svm.SVC(**params,probability=True)
#
#    ## fit model on training data
#    clf = clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    print(classification_report(y_test,y_pred))
#
#    ## retrain using all data
#    clf.fit(X, y)
#    print("... saving model: {}".format(SAVED_MODEL))
#    joblib.dump(clf,SAVED_MODEL)
#    runtime =convert_second(time.time() - time_start)
#    _create_training_log(X_train,runtime)
#
#def model_load():
#    """
#    example funtion to load model
#    """
#
#    if not os.path.exists(SAVED_MODEL):
#        raise Exception("Model '{}' cannot be found did you train the model?".format(SAVED_MODEL))
#    
#    model = joblib.load(SAVED_MODEL)
#    return(model)
#
#def model_predict(query,model=None):
#    """
#    example funtion to predict from model
#    """
#    time_start = time.time()
#    ## load model if needed
#    if not model:
#        model = model_load()
#    
#    ## output checking
#    if len(query.shape) == 1:
#        query = query.reshape(1, -1)
#    
#    ## make prediction and gather data for log entry
#    y_pred = model.predict(query)
#    y_proba = None
#    if 'predict_proba' in dir(model) and model.probability == True:
#        y_proba = model.predict_proba(query) 
#    runtime =convert_second(time.time() - time_start)
#    _update_predict_log(y_pred,y_proba,query,runtime)
#    return({'y_pred':y_pred,'y_proba':y_proba})

def convert_second(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds) 

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    ## train the model
    model_train()

    ## load the model
    model = model_load()
    
    ## example predict
    for query in [np.array([[6.1,2.8]]), np.array([[7.7,2.5]]), np.array([[5.8,3.8]])]:
        result = model_predict(query,model)
        y_pred = result['y_pred']
        print("predicted: {}".format(y_pred))



