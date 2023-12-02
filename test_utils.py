from utils import get_hyperparameter_combinations, train_test_dev_split,read_digits, tune_hparams, preprocess_data
from sklearn.linear_model import LogisticRegression
from joblib import load
import os
def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

def create_dummy_hyperparameter():
    gamma_list = [0.001, 0.01]
    C_list = [1]
    h_params={}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params)
    return h_params_combinations
def create_dummy_data():
    X, y = read_digits()
    
    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev = X[:50,:,:]
    y_dev = y[:50]

    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)

    return X_train, y_train, X_dev, y_dev
def test_for_hparam_cominations_values():    
    h_params_combinations = create_dummy_hyperparameter()
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)

def test_model_saving():
    X_train, y_train, X_dev, y_dev = create_dummy_data()
    h_params_combinations = create_dummy_hyperparameter()

    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)   

    assert os.path.exists(best_model_path)

def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .1
    dev_size = .6

    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert  ((len(X_dev) == 60))


# def test_solver_name_match():
#     model_file_name = 'models/m22aie228_lr_solver_lbfgs.joblib'
#     model = load(model_file_name)
#     model_solver = model.get_params()['solver']

#     # Extract the solver name from the file name
#     expected_solver = model_file_name.split('_')[2].split('.')[0]
#     assert (model_solver == expected_solver)


# def test_logistic_regression_model_type():
#     # Test case to check if the loaded model is a Logistic Regression model
#     model_filename = 'models/m22aie228_lr_solver_lbfgs.joblib'  # Update with your actual file path
#     if os.path.exists(model_filename):
#         model = load(model_filename)
#         assert isinstance(model, LogisticRegression)
#     else:
#         assert False, "Model file does not exist"


def test_logistic_regression_model_type():
    # Test case to check if the loaded model is a Logistic Regression model
    model_filename = 'models/m22aie228_lr_solver_lbfgs.joblib'  # Update with your actual file path
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert isinstance(model, LogisticRegression)
    else:
        assert False, "Model file does not exist"


    model_filename = 'models/m22aie228_lr_solver_liblinear.joblib'  # Update with your actual file path
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert isinstance(model, LogisticRegression)
    else:
        assert False, "Model file does not exist"


    model_filename = 'models/m22aie228_lr_solver_newton-cg.joblib'  # Update with your actual file path
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert isinstance(model, LogisticRegression)
    else:
        assert False, "Model file does not exist"


    model_filename = 'models/m22aie228_lr_solver_sag.joblib'  # Update with your actual file path
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert isinstance(model, LogisticRegression)
    else:
        assert False, "Model file does not exist"


    model_filename = 'models/m22aie228_lr_solver_saga.joblib'  # Update with your actual file path
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert isinstance(model, LogisticRegression)
    else:
        assert False, "Model file does not exist"                

def test_logistic_regression_solver():
    # Test case to check if the solver of the Logistic Regression model matches the expected solver
    model_filename = 'models/m22aie228_lr_solver_lbfgs.joblib'  # Update with your actual file path
    expected_solver = 'lbfgs'  # Update this to the correct solver name used in your model
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert model.get_params()['solver'] == expected_solver
    else:
        assert False, "Model file does not exist"        

   # Test case to check if the solver of the Logistic Regression model matches the expected solver
    model_filename = 'models/m22aie228_lr_solver_liblinear.joblib'  # Update with your actual file path
    expected_solver = 'liblinear'  # Update this to the correct solver name used in your model
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert model.get_params()['solver'] == expected_solver
    else:
        assert False, "Model file does not exist"     


    # Test case to check if the solver of the Logistic Regression model matches the expected solver
    model_filename = 'models/m22aie228_lr_solver_newton-cg.joblib'  # Update with your actual file path
    expected_solver = 'newton-cg'  # Update this to the correct solver name used in your model
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert model.get_params()['solver'] == expected_solver
    else:
        assert False, "Model file does not exist"     

    # Test case to check if the solver of the Logistic Regression model matches the expected solver
    model_filename = 'models/m22aie228_lr_solver_sag.joblib'  # Update with your actual file path
    expected_solver = 'sag'  # Update this to the correct solver name used in your model
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert model.get_params()['solver'] == expected_solver
    else:
        assert False, "Model file does not exist"     

    # Test case to check if the solver of the Logistic Regression model matches the expected solver
    model_filename = 'models/m22aie228_lr_solver_saga.joblib'  # Update with your actual file path
    expected_solver = 'saga'  # Update this to the correct solver name used in your model
    if os.path.exists(model_filename):
        model = load(model_filename)
        assert model.get_params()['solver'] == expected_solver
    else:
        assert False, "Model file does not exist"                  