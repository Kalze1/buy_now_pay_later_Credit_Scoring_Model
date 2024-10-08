import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from datetime import datetime


# Data Splitting Function
def split_data(df, target_column = "Label", test_size = 0.3, random_state =42):
    """Splits the dataframe into training and test sets."""
    y = df[target_column]
    x = df.drop(columns = [target_column])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= test_size, random_state= random_state)
    return x_train, x_test, y_train, y_test


# Model Training Function
def train_model(x_train, y_train, model_type = 'logistic'):
    """Trains a model based on the specified model type."""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'gbm':
        model = GradientBoostingClassifier()
    else:
        raise ValueError("Inbalid model type. Choose from ['logistic', 'decition_tree', 'random_forest', 'gbm']")
    

    model.fit(x_train, y_train)
    return model


# Hyperparameter Tuning Function
def tune_hyperparameters(x_train, y_train, model, param_grid, search_type='grid', cv=5, n_iter= 10):
    """Tunes the model's hyperparameters using Grid Search or Random Search."""
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter = n_iter, scoring= 'accuracy')
    else:
        raise ValueError("Invalid search type. Choose 'grid' or 'random'")
    
    search.fit(x_train, y_train)
    return search.best_estimator_


# Model Evaluation Function
def evaluate_model(model, x_test, y_test):
    """Evaluates the model's performance on the test data."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print("Classification Report:")
    print(report)


# Main function to run the model traing and evaluation
def run_model_training(df):
    """Runs the complet model selection, training, tuning, and evaluation pipeline."""
    # Split the data 
    x_train, x_test, y_train, y_test = split_data(df)

    #select two models for training
    models = ['logistic', 'random_forest']


    for model_type in models:
        print(f"\nTranining {model_type} model...")

        #Train the model
        model = train_model(x_train, y_train, model_type)
        

        #Define hyperparameter grid for tuning
        if model_type == 'logistic':
            param_grid ={'C':[0.01, 0.1, 1, 10]}
        elif model_type == 'random_forest':
            param_grid ={ 'n_estimators': [100, 200], 'max_depth': [10,20,30]}
        

        # Hyperparameter tuning
        best_model = tune_hyperparameters(x_train, y_train, model, param_grid, search_type = 'grid')
        

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Define the file name with the current date
        file_name = f'../data/credit_scoring_{model_type}_model_{current_date}.pkl'

        # Save the model to a file with the date in the name
        with open(file_name, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        



        # Evaluate the model 
        print(f"\nEvaluating {model_type} model after tuning...")
        evaluate_model(best_model, x_test, y_test)

