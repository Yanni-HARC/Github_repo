import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint
from joblib import dump, load
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ML_Algorithm:
    def __init__(self, model, outputfolder, random_state = 42):
        self.model = model
        self.outputfolder = outputfolder
        self.random_state = random_state

    def load_data(self, train_data_csv, test_data_csv, numerical_cols, categorical_cols, idx_col = 'Index', ref_col = 'Total_Life_Loss', filter_criteria = None, fill_nodata = None, save_data = True, name_convention = None):
        self.train_data_csv = train_data_csv
        self.test_data_csv = test_data_csv
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.idx_col = idx_col
        self.ref_col = ref_col

        self.X_train, self.y_train, self.indices_train, self.df_train = self.read_and_filter(self.train_data_csv, filter_criteria, fill_nodata, numerical_cols, categorical_cols, idx_col, ref_col)
        self.X_test, self.y_test, self.indices_test, self.df_test  = self.read_and_filter(self.test_data_csv, filter_criteria, fill_nodata, numerical_cols, categorical_cols, idx_col, ref_col)

        if save_data == True:
            self.name_convention = name_convention
            self.write_data()

    @staticmethod
    def read_and_filter(csv, filter_criteria, fill_nodata, numerical_cols, categorical_cols, idx_col, ref_col):
        # read data
        df = pd.read_csv(csv)

        # filter out rows if not relevant
        if filter_criteria == "structure":
            filtered_df = df.dropna(subset=['Structure_Stability_Criteria'])
        elif filter_criteria == "road":
            filtered_df = df.dropna(subset=['Vehicle_Type'])
        else:
            filtered_df = df

        # fill nodata value
        if fill_nodata is not None:
            filtered_df.fillna(fill_nodata, inplace=True)

        # combine numerical and categorical columns
        X_ = filtered_df[numerical_cols + categorical_cols]
        y_ = filtered_df[ref_col]
        indices = filtered_df[idx_col]

        return X_, y_, indices, filtered_df

    def setup_pipeline(self,):
        if self.model == 'HistGradientBoosting_Regressor':
            # apply normalization for numerical columns
            preprocessor = ColumnTransformer(transformers=[
                ('num', MinMaxScaler(), self.numerical_cols),
                # 'passthrough' for categorical features if no preprocessing is needed
                ('cat', 'passthrough', self.categorical_cols)
            ])
            # set up the pipeline
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('gbr', HistGradientBoostingRegressor(random_state=self.random_state))
            ])   

            return pipe
                
        elif self.model == 'Linear':
        # apply normalization for numerical columns, enforce one hot encoder for categorical columns
            preprocessor = ColumnTransformer(transformers=[
                ('num', MinMaxScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])
            # set up the pipeline
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('lr', LinearRegression())
            ])        

            return pipe
    
    def perform_grid_search(self, kf_config = None, save_best_model = True, save_cv_results = False):       
        # apply kfold validation 
        if kf_config == None:
            kf_config ={
                'n_splits': 5, 
                'shuffle': True, 
                'random_state': self.random_state
            }
        kf = KFold(**kf_config)

        # set up pipe
        pipe = self.setup_pipeline()
            
        # hyperparameter fine-tuning
        if self.model == 'HistGradientBoosting_Regressor':
            param_grid = {
                # 'gbr__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Often has a big impact, so we keep a few distinct values
                # 'gbr__max_iter': [100, 200, 300],  # Controls the number of trees, similar to n_estimators
                # 'gbr__max_depth': [None, 3, 5, 10],  # A good range to control the depth of each tree
                # 'gbr__min_samples_leaf': [20, 30, 40],  # Helps control over-fitting
                # 'gbr__l2_regularization': [0.0, 0.01, 0.1, 1.0],  # Regularization can help with generalization
                # 'gbr__max_leaf_nodes': [None, 31, 41, 51],  # Control the size of the trees

                'gbr__learning_rate': [0.05, 0.1],  # Reduced to two distinct values
                'gbr__max_iter': [100, 200],  # Reduced to two values
                'gbr__max_depth': [None, 3, 5],  # Reduced to three values
                'gbr__min_samples_leaf': [20, 30],  # Reduced to two values
            }

        elif self.model == 'Linear':
            param_grid = {
                'lr__fit_intercept': [True, False],  # Whether to calculate the intercept for this model
                'lr__copy_X': [True, False],  # If True, X will be copied; else, it may be overwritten.
                'lr__n_jobs': [-1],  # Number of CPU cores used when parallelizing over classes
                'lr__positive': [True, False]  # Forces the coefficients to be positive. Available from version 0.24.
            }

        # fit model with grid search, and use r2 as scoring       
        grid_search = GridSearchCV(pipe, param_grid, cv=kf, scoring='r2', verbose=3, error_score='raise')
        grid_search.fit(self.X_train, self.y_train) 
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_}")

        if save_best_model == True:
            self.write_best_model(kf_config, grid_search.best_estimator_)

        if save_cv_results == True:
            self.write_cv_results(grid_search.cv_results_)
            
    def save_Xy(self, X, y, train_or_test):
        dump(X, f'{self.outputfolder}{self.name_convention}_X_{train_or_test}.joblib')
        dump(y, f'{self.outputfolder}{self.name_convention}_y_{train_or_test}.joblib')

    def write_data(self):
        self.df_train.to_csv(f'{self.outputfolder}{self.name_convention}_train.csv', index= False)
        self.df_test.to_csv(f'{self.outputfolder}{self.name_convention}_test.csv', index= False)

        self.save_Xy(self.X_train, self.y_train, 'train')
        self.save_Xy(self.X_test, self.y_test, 'test')

        dump(self.numerical_cols, f'{self.outputfolder}{self.name_convention}_numerical_col_lst.joblib')
        dump(self.categorical_cols, f'{self.outputfolder}{self.name_convention}_categorical_col_lst.joblib')
        dump(self.numerical_cols + self.categorical_cols, f'{self.outputfolder}{self.name_convention}_feature_col_lst.joblib')

    def write_cv_results(self, cv_results, extra_notation = ''):
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(f'{self.outputfolder}{self.name_convention}_cv_results{extra_notation}.csv', index=False)

    def write_best_model(self, kf_config, best_model, extra_notation = ''):
        with open(f'{self.outputfolder}{self.name_convention}_kf_config.json', 'w') as f:
            json.dump(kf_config, f)         

        dump(best_model, f'{self.outputfolder}{self.name_convention}_best_model{extra_notation}.joblib')
        print(f"Best model saved to: {self.outputfolder}{self.name_convention}_best_model{extra_notation}.joblib")

    def write_prediction(self, y_pred, train_or_test):
        if train_or_test == 'train':
            df = self.df_train
            indice = self.indices_train
        elif train_or_test == 'test':
            df = self.df_test
            indice = self.indices_test

        prediction_df = pd.DataFrame({self.idx_col: indice, 'Pred_Total_Life_Loss': y_pred})
        merged_df = pd.merge(df, prediction_df, on=self.idx_col, how='inner')
        merged_df.to_csv(self.outputfolder + f'{self.name_convention}_{train_or_test}.csv', index=False)
        print(f"Prediction saved to: {self.outputfolder}{self.name_convention}_{train_or_test}.csv")
           
    @staticmethod
    def evaluate_performance(y, y_pred):
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        syt = np.sum(y)
        syp = np.sum(y_pred)
        pse = (syp - syt)/(syt)

        print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        print(f"Sum y_true: {syt}, Sum y_pred: {syp}, Percentage Sum Error: {pse}")

        return mae, rmse, r2, syt, syp, pse
                
    def predict(self, round_integer = True, save_csv = True, save_log = True):
        # use the best model for prediction
        best_model = load(f'{self.outputfolder}{self.name_convention}_best_model.joblib')
        y_pred_train = best_model.predict(self.X_train)
        y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)

        y_pred_test = best_model.predict(self.X_test)
        y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)

        # if round prediction to integer
        if round_integer == True:
            y_pred_train = [round(num) for num in y_pred_train]
            y_pred_test = [round(num) for num in y_pred_test]

        # evaluate performance and save log
        if save_csv == True:
            self.write_prediction(y_pred_train, train_or_test='train')
            self.write_prediction(y_pred_test, train_or_test='test')

        if save_log == True:
            print("Train Performance Matrix")
            train_mae, train_rmse, train_r2, train_syt, train_syp, train_pse = self.evaluate_performance(self.y_train, y_pred_train)
            print("Test Performance Matrix")
            test_mae, test_rmse, test_r2, test_syt, test_syp, test_pse = self.evaluate_performance(self.y_test, y_pred_test)

            self.write_results_log(train_mae, train_rmse, train_r2, train_syt, train_syp, train_pse, test_mae, test_rmse, test_r2, test_syt, test_syp, test_pse)
    
    def write_results_log(self, train_mae, train_rmse, train_r2, train_syt, train_syp, train_pse, test_rmae, test_mse, test_r2, test_syt, test_syp, test_pse):
        with open(self.outputfolder + f'{self.name_convention}_results_log.txt', 'w') as file:
            file.write(f"## Run Info")
            file.write(f"\nTrain Data: {self.train_data_csv}")
            file.write(f"\nTest Data: {self.test_data_csv}")
            file.write(f"\nSelected Algorithm: {self.model}")
            file.write(f"\n## Grid Search")   
            file.write(f"\nBest model: {self.outputfolder}{self.name_convention}_best_model.joblib")

            file.write(f"\n## Train Performance Matrix")
            file.write(f"\nMean Absolute Error: {train_mae}")
            file.write(f"\nRoot Mean Squared Error: {train_rmse}")
            file.write(f"\nR^2 Score: {train_r2}")
            file.write(f"\nSum y_true: {train_syt}")
            file.write(f"\nSum y_pred: {train_syp}")
            file.write(f"\nPercentage Sum Error: {train_pse}")    

            file.write(f"\n## Test Performance Matrix")
            file.write(f"\nMean Absolute Error: {test_rmae}")
            file.write(f"\nRoot Mean Squared Error: {test_mse}")
            file.write(f"\nR^2 Score: {test_r2}")
            file.write(f"\nSum y_true: {test_syt}")
            file.write(f"\nSum y_pred: {test_syp}")
            file.write(f"\nPercentage Sum Error: {test_pse}")   
