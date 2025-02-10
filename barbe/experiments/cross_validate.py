import itertools
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from dill import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def cross_validate_nn(training_data, training_labels, model_name="neural", use_categorical=None):
    x_train, x_test, y_train, y_test = train_test_split(training_data, training_labels, test_size=0.2)

    if use_categorical is not None:
        encoder = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        preprocess = ColumnTransformer([('enc', encoder, use_categorical)], remainder='passthrough')
    hidden_layer_sizes = [(5, 2,), (5, 5,), (10, 5,),
                          (10, 10,), (50, 10,), (50, 50,),
                          (50, 10, 10,), (50, 50, 10,)]
    activation_functions = ['logistic', 'tanh', 'relu']
    solver_types = ['lbfgs', 'sgd', 'adam']
    alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    best_performance = 0
    best_model_settings = None
    for hidden_layer, activation, solver, alpha in itertools.product(hidden_layer_sizes,
                                                                     activation_functions,
                                                                     solver_types,
                                                                     alpha_values):
        print("Running Tests: ", hidden_layer, activation, solver, alpha)
        if use_categorical is not None:
            model = Pipeline([('pre', preprocess),
                              ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer,
                                                    activation=activation,
                                                    solver=solver,
                                                    alpha=alpha,
                                                    early_stopping=True,
                                                    tol=1e-5))])
        else:
            model = MLPClassifier(hidden_layer_sizes=hidden_layer,
                                  activation=activation,
                                  solver=solver,
                                  alpha=alpha,
                                  early_stopping=True,
                                  tol=1e-5)
        model.fit(x_train, y_train)
        temp_performance = model.score(x_test, y_test)
        print("Current Performance Original - Validation: ", model.score(x_train, y_train), " - ", temp_performance)
        if temp_performance > best_performance:
            best_performance = temp_performance
            best_model_settings = (hidden_layer, activation, solver, alpha)

    hidden_layer, activation, solver, alpha = best_model_settings
    if use_categorical is not None:
        model = Pipeline([('pre', preprocess),
                          ('clf', MLPClassifier(hidden_layer_sizes=hidden_layer,
                                                activation=activation,
                                                solver=solver,
                                                alpha=alpha,
                                                early_stopping=True,
                                                tol=1e-5))])
    else:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer,
                              activation=activation,
                              solver=solver,
                              alpha=alpha,
                              early_stopping=True,
                              tol=1e-5)
    model.fit(training_data, training_labels)
    print("Best Settings: ", best_model_settings)
    print("Best Performance: ", best_performance)

    with open("./"+model_name+'.pickle', 'wb') as f:
        dump(model, f)
