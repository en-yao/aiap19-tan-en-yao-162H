import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, recall_score,
                             confusion_matrix, precision_score, f1_score, auc, roc_curve, roc_auc_score)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import config


class ModelEvaluation:
    def __init__(self, config_file):
        # Load the configuration file
        self.config = config

        # Set random seed for reproducibility
        self.random_state = int(self.config.split['random_state'])

        folder_path = self.config.file_paths['plot_folder']

        # Ensure the folder exists; if not, create it
        if not os.path.exists(folder_path):
            # If the folder doesn't exist, create it
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")


        # Connect to the database and read the data into a DataFrame
        db_path = self.config.database['path']
        table_name = self.config.database['table']

        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        self.data = pd.read_sql(query, conn)
        conn.close()

        # Data processing step (handle missing values, encode, etc.)
        self.process_data()

        # Define the features to exclude for regression
        regression_excluded_columns = [
            'System Location Code_Zone_B',
            'System Location Code_Zone_C',
            'System Location Code_Zone_D',
            'System Location Code_Zone_E',
            'System Location Code_Zone_F',
            'System Location Code_Zone_G',
            'Plant Type-Stage Encoded'
        ]

        # Create a copy of self.data to preserve the original data
        data_copy = self.data.copy()

        # Initialize features and targets for regression and classification
        self.X_regression = data_copy.drop(columns=[self.config.targets['temperature']] + regression_excluded_columns)
        self.X_classification = data_copy.drop(columns=self.config.targets['plant_type_stage'])
        
        self.y_regression = self.data[self.config.targets['temperature']]
        self.y_classification = self.data[self.config.targets['plant_type_stage']]

        # First, split for regression task
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
            self.X_regression, self.y_regression, 
            test_size=float(self.config.split['test_size']), 
            random_state=self.random_state
        )

        # Then, split for classification task
        self.X_train_cls, self.X_test_cls, self.y_train_cls, self.y_test_cls = train_test_split(
            self.X_classification, self.y_classification, 
            test_size=float(self.config.split['test_size']), 
            random_state=self.random_state
        )

        # Initialize models for both regression and classification
        self.models = {
            'RandomForestRegressor': RandomForestRegressor(random_state=self.random_state),
            'RandomForestClassifier': RandomForestClassifier(random_state=self.random_state),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=self.random_state),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=self.random_state),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=self.random_state),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=self.random_state),
        }

        # Hyperparameter grids for both regression and classification
        self.param_grids = {
            'RandomForestRegressor': self.config.param_grid['RandomForestRegressor'],
            'RandomForestClassifier': self.config.param_grid['RandomForestClassifier'],
            'DecisionTreeRegressor': self.config.param_grid['DecisionTreeRegressor'],
            'DecisionTreeClassifier': self.config.param_grid['DecisionTreeClassifier'],
            'GradientBoostingRegressor': self.config.param_grid['GradientBoostingRegressor'],
            'GradientBoostingClassifier': self.config.param_grid['GradientBoostingClassifier']
        }

    def process_data(self):
        # Convert nutrient sensor data to numerical format
        self.convert_nutrient_sensor_data()

        # Address and correct data errors
        self.correct_data_errors()

        # Standardize text case in categorical columns
        self.standardise_case_variants()

        # Generate the 'Plant Type-Stage' feature
        self.create_plant_type_stage()

        # Handle missing data through imputation
        self.handle_missing_values()

        # Remove duplicate records from the dataset
        self.remove_duplicate_rows()

        # Encode categorical variables after removing duplicates
        self.encode_data()

        self.drop_redundant_features()
        # Additional processing like encoding categorical variables or handling outliers can go here


    def convert_nutrient_sensor_data(self):
        # Check if the required nutrient sensor columns exist before processing
        required_columns = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            print(f"Warning: The following columns are missing from the data: {', '.join(missing_columns)}")
        else:
            # Convert the nutrient sensor columns to numeric values
            for col in required_columns:
                # Extract numeric values from the string data
                self.data[col] = self.data[col].str.extract('(\d+)')
                # Convert the extracted values to numeric, handling any errors by coercing to NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(float)

    def correct_data_errors(self):
        # Check if required columns exist before filtering data
        required_columns = ['Temperature Sensor (°C)', 'Light Intensity Sensor (lux)', 'EC Sensor (dS/m)']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            print(f"Warning: The following columns are missing from the data: {', '.join(missing_columns)}")
        else:
            # Filter out rows with negative values in the required sensor columns
            self.data = self.data[
                (self.data['Temperature Sensor (°C)'] >= 0) &
                (self.data['Light Intensity Sensor (lux)'] >= 0) &
                (self.data['EC Sensor (dS/m)'] >= 0)
            ]
    def standardise_case_variants(self):
        # Standardise the text case in the specified columns to title case
        for col in ['Plant Type', 'Plant Stage']:
            if col in self.data.columns:
                self.data[col] = self.data[col].str.title()
            else:
                print(f"Warning: Column '{col}' not found in the data.")

    def create_plant_type_stage(self):
            # Create a combined 'Plant Type-Stage' column manually
            if 'Plant Type' in self.data.columns and 'Plant Stage' in self.data.columns:
                self.data['Plant Type-Stage'] = self.data['Plant Type'] + '-' + self.data['Plant Stage']
            else:
                # Handle case where columns are missing, or implement other logic for creation
                print("Warning: Missing 'Plant Type' or 'Plant Stage' columns.")
                # self.data['Plant Type-Stage'] = 'Unknown'  # Default or fallback value

    def handle_missing_values(self):
        # Check if the required columns exist before performing imputation
        required_columns = ['Humidity Sensor (%)', 'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)', 'Water Level Sensor (mm)']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            print(f"Warning: The following columns are missing from the data: {', '.join(missing_columns)}")
            return  # Exit the function if any required columns are missing
        
        # Impute missing values for larger portion using RandomForestRegressor
        observed = self.data[self.data['Humidity Sensor (%)'].notna()]
        missing = self.data[self.data['Humidity Sensor (%)'].isna()]

        rf = RandomForestRegressor(n_estimators=200, random_state=0)
        rf.fit(observed.index.values.reshape(-1, 1), observed['Humidity Sensor (%)'])

        self.data.loc[self.data['Humidity Sensor (%)'].isna(), 'Humidity Sensor (%)'] = rf.predict(missing.index.values.reshape(-1, 1))

        # Impute missing values for smaller portion using KNNImputer
        columns_to_impute = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)', 'Water Level Sensor (mm)']

        imputer = KNNImputer(n_neighbors=5)
        self.data[columns_to_impute] = imputer.fit_transform(self.data[columns_to_impute])

    def remove_duplicate_rows(self):
        # Remove duplicate rows from the dataset
        self.data = self.data.drop_duplicates()

    def encode_data(self):
        # Check if 'System Location Code' column exists before encoding
        if 'System Location Code' in self.data.columns and 'Previous Cycle Plant Type' in self.data.columns:
            self.data = pd.get_dummies(self.data, columns=['System Location Code', 'Previous Cycle Plant Type'], drop_first=True)

        else:
            print("Warning: Column 'System Location Code' not found in the data.")

        # Check if 'Plant Type-Stage' column exists before encoding
        if 'Plant Type-Stage' in self.data.columns:
            encoder = LabelEncoder()
            self.data['Plant Type-Stage Encoded'] = encoder.fit_transform(self.data['Plant Type-Stage'])

            # Save the encoder to access the class names later
            self.label_encoder = encoder

        else:
            print("Warning: Column 'Plant Type-Stage' not found in the data.")

    def drop_redundant_features(self):
        # Remove columns that are redundant for the analysis.
        redundant_columns = ['Plant Type', 'Plant Stage', 'Plant Type-Stage']
        self.data.drop(columns=redundant_columns, inplace=True)

    def perform_random_search(self, model, param_grid, X, y, cv):
        """Perform RandomizedSearchCV to find the best hyperparameters."""
        # Determine the scoring metric based on the type of model
        if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor)):
            scoring_metric = 'neg_mean_squared_error'
        else:
            scoring_metric = 'accuracy'

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=int(self.config.grid_search['n_iter']),
            cv=cv,
            scoring=scoring_metric,
            random_state=self.random_state,
            n_jobs=-1
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_

    def evaluate_regression_model(self, model, model_name):
        # Perform hyperparameter tuning
        if model_name in self.param_grids:
            print(f"Tuning hyperparameters for {model_name}...")
            model, best_params = self.perform_random_search(
                model, self.param_grids[model_name], self.X_train_reg, self.y_train_reg, int(self.config.grid_search['cv'])
            )
            print(f"Best parameters for {model_name}: {best_params}")
    
        # Make predictions
        y_pred = model.predict(self.X_test_reg)
        rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
        r2 = r2_score(self.y_test_reg, y_pred)

        print(f'{model_name} - RMSE: {rmse}')
        print(f'{model_name} - R²: {r2}')

        # Plot predicted vs actual
        self.plot_predicted_vs_actual(self.y_test_reg, y_pred, model_name)

        # Plot CV scores and feature importance
        cv_scores = cross_val_score(model, self.X_train_reg, self.y_train_reg, cv=int(self.config.grid_search['cv']))
        self.plot_cv_scores(cv_scores, model_name)
        
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, model_name)

    def evaluate_classification_model(self, model, model_name):
        # Perform hyperparameter tuning
        if model_name in self.param_grids:
            print(f"Tuning hyperparameters for {model_name}...")
            model, best_params = self.perform_random_search(
                model, self.param_grids[model_name], self.X_train_cls, self.y_train_cls, int(self.config.grid_search['cv'])
            )
            print(f"Best parameters for {model_name}: {best_params}")
        
        # Make predictions
        y_pred = model.predict(self.X_test_cls)
        
        accuracy = accuracy_score(self.y_test_cls, y_pred)
        precision = precision_score(self.y_test_cls, y_pred, average='weighted')
        recall = recall_score(self.y_test_cls, y_pred, average='weighted')

        print(f'{model_name} - Accuracy: {accuracy}')
        print(f'{model_name} - Precision: {precision}')
        print(f'{model_name} - Recall: {recall}')

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test_cls, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Ensure labels don't get cut off at the boundaries
        plt.tight_layout()

        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # Plot CV scores and feature importance
        cv_scores = cross_val_score(model, self.X_train_cls, self.y_train_cls, cv=int(self.config.grid_search['cv']))
        self.plot_cv_scores(cv_scores, model_name)
        
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, model_name)

        # Plot AUC-ROC curve
        self.plot_roc_curve(model, self.X_test_cls, self.y_test_cls, model_name)

    def plot_cv_scores(self, cv_scores, model_name):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=cv_scores, width=0.4, linewidth=1.5, color="skyblue")

        # Define consistent limits
        y_min = 0
        y_max = 1

        # Standardize y-axis limits
        plt.ylim(y_min, y_max)

        plt.title(f'Cross-validation Scores for {model_name}')
        plt.ylabel('Score')
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'{model_name}_cv_scores.png'))
        plt.close()

    def plot_feature_importance(self, model, model_name):
        feature_importance = model.feature_importances_
        if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor)):
            features = self.X_regression.columns
        elif isinstance(model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier)):
            features = self.X_classification.columns
        # features = self.X.columns
        plt.figure(figsize=(10, 6))
        plt.barh(features, feature_importance)
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Importance')

        # Adjust layout to avoid y-labels getting cut off
        plt.tight_layout()

        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'{model_name}_feature_importance.png'))
        plt.close()

    def plot_roc_curve(self, model, X_test, y_test, model_name):
        # Get probabilities for all classes
        y_prob = model.predict_proba(X_test)

        # Initialize plot
        plt.figure(figsize=(10, 6))

        # Get the class labels (names)
        n_classes = y_prob.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'C{self.label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')

        # Plot diagonal line (no-skill classifier)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

        # Set title and labels
        plt.title(f'AUC-ROC Curve for {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # Add legend
        plt.legend(loc='lower right')

        # Add legend with padding adjustments between the lines and labels
        # plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0.5), labelspacing=1.5, handlelength=3, handleheight=1.5)

        # Save the plot
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'{model_name}_auc_roc.png'))
        plt.close()

    def plot_predicted_vs_actual(self, y_actual, y_pred, model_name):
        # Initialize plot
        plt.figure(figsize=(8, 6))
        
        # Create scatter plot
        sns.scatterplot(x=y_actual, y=y_pred, color='blue', edgecolor='black', label='Predicted vs Actual')
        
        # Plot a diagonal line (perfect prediction line)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual for {model_name}')
        
        # Add legend
        plt.legend(loc='upper left')

        # Save the plot
        plt.savefig(os.path.join(self.config.file_paths['plot_folder'], f'{model_name}_pred_vs_actual.png'))
        plt.close()

    def run_evaluation(self):
        # Main evaluation loop
        for model_name, model in self.models.items():
            if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor)):
                self.evaluate_regression_model(model, model_name)
            elif isinstance(model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier)):
                self.evaluate_classification_model(model, model_name)





