"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
# 2.1. SVM
gamma_list = [0.001, 0.01, 0.1, 1]
C_list = [1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list
h_params_combinations = get_hyperparameter_combinations(h_params)

test_sizes =  [0.1, 0.2, 0.3, 0.45]
dev_sizes  =  [0.1, 0.2, 0.3, 0.45]
for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1- test_size - dev_size
        # 3. Data splitting -- to create train and test sets                
        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
        # 4. Data preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)
    
        best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)        
    
        # loading of model         
        best_model = load(best_model_path) 

        test_acc = predict_and_eval(best_model, X_test, y_test)
        train_acc = predict_and_eval(best_model, X_train, y_train)
        dev_acc = best_accuracy

        print("test_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(test_size, dev_size, train_size, train_acc, dev_acc, test_acc))
=======
from sklearn import datasets
from utils import tune_hparams, split_train_dev_test, train_model, predict_and_eval, rebuild_classification_report_from_cm, param_combinations 

# Data specific to a given problem
def load_data():
    return datasets.load_digits()

def preprocess_data(images):
    n_samples = len(images)
    print("Number of train, test, and dev samples together =", n_samples)
    
    height, width = images[0].shape
    print("Size of the images in the dataset: Height =", height, ", Width =", width)
    
    return images.reshape((n_samples, -1))

# 1. Load data
digits = load_data()

# 2. Plot some samples
#plot_samples(digits.images, digits.target, title="Training")

# 3. Preprocess data
data = preprocess_data(digits.images)


gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
# Create a combination of all parameters
list_of_all_param_combinations = param_combinations(gamma_ranges, C_ranges)

# Splitting
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        X_Train, X_Test, Y_Train, Y_Test, X_Dev, Y_Dev = split_train_dev_test(data, digits.target, test_size, dev_size)
        best_hparams, best_model, best_accuracy_dev = tune_hparams(X_Train, Y_Train, X_Dev, Y_Dev, list_of_all_param_combinations)

        # Prediction and performance on train set
        best_accuracy_train = predict_and_eval(best_model, X_Train, Y_Train)

        # Prediction and performance on test set
        best_accuracy_test = predict_and_eval(best_model, X_Test, Y_Test)

        # 7. Output the results
        #print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size:.1f} train_acc={best_accuracy_train} dev_acc={best_accuracy_dev:.3f} test_acc={best_accuracy_test:.3f}")
        #print(f"best_hparams: {best_hparams}")

# 4. Split data
# X_Train, X_Test, Y_Train, Y_Test, X_Dev, Y_Dev = split_train_dev_test(data, digits.target, 0.5, 0.1)
# X_Train, X_Test, Y_Train, Y_Test = split_train_dev_test(data, digits.target, 0.5, 0)

# 5. Train model
# model = train_model(X_Train, Y_Train)

# 6. Predict & Evaluate
# predicted = predict_and_eval(model, X_Test, Y_Test)

# 7. Rebuild and display classification report from confusion matrix
# rebuild_classification_report_from_cm(Y_Test, predicted)

