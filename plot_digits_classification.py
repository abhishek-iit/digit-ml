from sklearn import datasets
from utils import split_train_dev_test, train_model, predict_and_eval, rebuild_classification_report_from_cm 

# Data specific to a given problem
def load_data():
    return datasets.load_digits()

def preprocess_data(images):
    n_samples = len(images)
    return images.reshape((n_samples, -1))

# 1. Load data
digits = load_data()

# 2. Plot some samples
#plot_samples(digits.images, digits.target, title="Training")

# 3. Preprocess data
data = preprocess_data(digits.images)

# 4. Split data
#X_Train, X_Test, Y_Train, Y_Test, X_Dev, Y_Dev = split_train_dev_test(data, digits.target, 0.5, 0.5)
X_Train, X_Test, Y_Train, Y_Test = split_train_dev_test(data, digits.target, 0.5, 0)

# 5. Train model
clf = train_model(X_Train, Y_Train)

# 6. Predict & Evaluate
predicted = predict_and_eval(clf, X_Test, Y_Test)

# 7. Rebuild and display classification report from confusion matrix
rebuild_classification_report_from_cm(Y_Test, predicted)

