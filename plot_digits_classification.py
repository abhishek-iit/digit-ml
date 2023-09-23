import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.transform import resize

# Load data
digits = datasets.load_digits()

# Resize function
def resize_images(images, output_shape):
    return np.array([resize(image, output_shape, mode='reflect', anti_aliasing=True) for image in images])

# Split and train function
def train_and_evaluate(images, labels, train_size=0.7, dev_size=0.1, test_size=0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, train_size=train_size, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, train_size=dev_size/(dev_size+test_size), random_state=42)
    
    # Flatten the images for training
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_dev = X_dev.reshape(X_dev.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Train a classifier
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train, y_train) * 100
    dev_acc = clf.score(X_dev, y_dev) * 100
    test_acc = clf.score(X_test, y_test) * 100
    
    return train_acc, dev_acc, test_acc

# Main function
def main():
    for size in [(4, 4), (6, 6), (8, 8)]:
        resized_images = resize_images(digits.images, size)
        train_acc, dev_acc, test_acc = train_and_evaluate(resized_images, digits.target)
        print(f"image size: {size[0]}x{size[1]} train_size: 0.7 dev_size: 0.1 test_size: 0.2 train_acc: {train_acc:.1f} dev_acc: {dev_acc:.1f} test_acc: {test_acc:.1f}")

if __name__ == "__main__":
    main()
