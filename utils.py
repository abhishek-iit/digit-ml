import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import product


""" Functions common to all classification problems """
    
# Splits the given dataset into a set of training, development/validation and testing data
def split_train_dev_test(data, labels, test_size, dev_size):
    X_Train_Dev, X_Test, Y_Train_Dev, Y_Test = train_test_split(
        data, labels, test_size=test_size, random_state=1)
    
    if dev_size > 0:
        # Calculate train size given the dev_size
        train_size = dev_size/(1-test_size)
        
        X_Train, X_Dev, Y_Train, Y_Dev = train_test_split(
            X_Train_Dev, Y_Train_Dev, train_size=train_size,random_state=1)
        
        return X_Train, X_Dev, Y_Train, Y_Dev, X_Test, Y_Test

    return X_Train_Dev, X_Test, Y_Train_Dev, Y_Test


# Train a machine learning model.
# As need arises add new model to the logic
# Parameters:
# - X: Features
# - y: Labels
# - model_params: Dictionary of model parameters
# - model_type: Type of model to train ("svm", "random_forest", "logistic_regression")

# Returns:
# Trained model

def train_model(X, y, model_params={"gamma": 0.001}, model_type="svm"):
    
    if model_params is None:
        model_params = {}
    
    # Initialize classifier based on model_type
    if model_type == "svm":
        clf = svm.SVC
    elif model_type == "random_forest":
        clf = RandomForestClassifier
    elif model_type == "logistic_regression":
        clf = LogisticRegression
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    # Create and train the model
    model = clf(**model_params)
    model.fit(X, y)
    
    return model

def predict(clf, X_test):
    return clf.predict(X_test)

def evaluate_model(y_true, y_pred, clf):
    # print(f"Classification report for classifier {clf}:\n"
    #       f"{metrics.classification_report(y_true, y_pred)}\n")
    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    
    # print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return metrics.accuracy_score(y_true, y_pred)
   

def predict_and_eval(model, X_test, y_test):
    # Predict
    predicted = predict(model, X_test)
  
    # Evaluate model
    return evaluate_model(y_test, predicted, model)   

def rebuild_classification_report_from_cm(y_test, predicted):
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    y_true = []
    y_pred = []
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")   
    
def param_combinations(gamma_ranges, C_ranges):
    # Create a combination of all parameters
    list_of_all_param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in product(gamma_ranges, C_ranges)]
    return list_of_all_param_combinations    

def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combinations):
    best_accuracy = -1
    best_hparams = None
    best_model = None

    for params in list_of_all_param_combinations:
        
        #print(params)
        model = train_model(X_train, y_train, params)
        accuracy_dev = predict_and_eval(model, X_dev, y_dev)
        if accuracy_dev > best_accuracy:
            best_accuracy = accuracy_dev
            best_hparams = params
            best_model = model

    return best_hparams, best_model, best_accuracy     