# config.py

# Database and table information
database = {
    'path': './data/agri.db',
    'table': 'farm_data'
}

# File paths for additional data
file_paths = {
    'plot_folder': './plot_folder'
}

# Targets to predict
targets = {
    'temperature': 'Temperature Sensor (°C)',
    'plant_type_stage': 'Plant Type-Stage Encoded'
}

# Split configurations
split = {
    'test_size': 0.2,
    'random_state': 42
}

# Parameter grids for classifiers
param_grid = {
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Grid search configuration
grid_search = {
    'cv': 5,
    'n_iter': 6
}
 
