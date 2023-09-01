import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_data():
    return datasets.load_digits()

def plot_samples(images, labels, title, n_samples=4):
    _, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    for ax, image, label in zip(axes, images[:n_samples], labels[:n_samples]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title}: {label}")

def preprocess_data(images):
    n_samples = len(images)
    return images.reshape((n_samples, -1))

def split_data(data, labels, test_size=0.5):
    return train_test_split(data, labels, test_size=test_size, shuffle=False)

def split_train_dev_test(X, y, test_size=0.2, dev_size=0.2):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    adjusted_dev_size = dev_size / (1 - test_size)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=adjusted_dev_size, shuffle=True)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def train_model(X_train, y_train, gamma=0.001):
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    return clf

def predict(clf, X_test):
    return clf.predict(X_test)

def evaluate_model(y_true, y_pred, clf):
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

def predict_and_eval(model, X_test, y_test):
    # 6. Predict
    predicted = predict(clf, X_test)
  
    # 7. Plot some test samples with predictions
    plot_samples(X_test.reshape(-1, 8, 8), predicted, title="Prediction")
  
    # 8. Evaluate model
    evaluate_model(y_test, predicted, clf)

    return predicted   

def rebuild_classification_report_from_cm(confusion_matrix):
    y_true = []
    y_pred = []
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")    
    
if __name__ == "__main__":
    # 1. Load data
    digits = load_data()
  
    # 2. Plot some samples
    plot_samples(digits.images, digits.target, title="Training")
  
    # 3. Preprocess data
    data = preprocess_data(digits.images)
  
    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(data, digits.target)
  
    # 5. Train model
    clf = train_model(X_train, y_train)
  
    # 6. Predict
    predicted = predict_and_eval(clf, X_test, y_test)

    # 9. Rebuild and display classification report from confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    rebuild_classification_report_from_cm(cm)
