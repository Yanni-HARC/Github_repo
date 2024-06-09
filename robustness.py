import numpy as np 
import pandas as pd
from sklearn import metrics
import warnings
from joblib import load
warnings.filterwarnings("ignore")

class Validation:
    def __init__(self, modelfolder, model_name_convention):
        self.modelfolder = modelfolder
        self.model_name_convention = model_name_convention
        self.load_model()

    def load_model(self):
        self.best_model = load(f'{self.modelfolder}{self.model_name_convention}_best_model.joblib')
        self.numerical_columns = load(f'{self.modelfolder}{self.model_name_convention}_numerical_col_lst.joblib')
        self.categorical_columns = load(f'{self.modelfolder}{self.model_name_convention}_categorical_col_lst.joblib')

    
    def load_data(self, validation_data_csv, outputfolder, filter_criteria, fill_nodata, numerical_cols, categorical_cols, 
                  idx_col = 'Index', ref_col = 'Total_Life_Loss', save_data = True, validation_name_convention = None):
        self.validation_data_csv = validation_data_csv
        self.outputfolder = outputfolder
        self.validation_name_convention = validation_name_convention        
        self.idx_col = idx_col
        self.ref_col = ref_col        

        df = pd.read_csv(self.validation_data_csv)
        for i in self.categorical_columns:
            df[i] = df[i].astype('category')

        self.x_val, self.y_val, self.indices, self.df = self.read_and_filter(self.validation_data_csv, filter_criteria, fill_nodata, numerical_cols, categorical_cols, idx_col, ref_col)

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
    
    @staticmethod
    def evaluate_performance(y_test, y_pred):
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        syt = np.sum(y_test)
        syp = np.sum(y_pred)
        pse = (syp - syt)/(syt)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
        auc = metrics.auc(fpr, tpr)

        print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}, AUC: {auc}")
        print(f"Sum y_true: {syt}, Sum y_pred: {syp}, Percentage Sum Error: {pse}")

        return mae, rmse, r2, syt, syp, pse

    def write_csv(self, y_pred):
        validation_df = pd.DataFrame({self.idx_col: self.indices, 'Pred_Total_Life_Loss': y_pred})
        merged_df = pd.merge(self.df, validation_df, on=self.idx_col, how='inner')
        merged_df.to_csv(self.outputfolder + f'{self.validation_name_convention}_validation.csv', index=False)

    def write_results_log(self, mae, rmse, r2, syt, syp, pse):
        with open(self.outputfolder + f'{self.validation_name_convention}_results_log.txt', 'w') as file:
            file.write(f"## Run Info")
            file.write(f"\nValidation Data: {self.validation_data_csv}")
            file.write(f"\nBest model: {self.outputfolder}{self.model_name_convention}_best_model.joblib")
            file.write(f'\nValidation results: {self.outputfolder}{self.validation_name_convention}_validation.csv')
            file.write(f"\n## Performance Matrix")
            file.write(f"\nMean Absolute Error: {mae}")
            file.write(f"\nRoot Mean Squared Error: {rmse}")
            file.write(f"\nR^2 Score: {r2}")
            file.write(f"\nSum y_true: {syt}")
            file.write(f"\nSum y_pred: {syp}")
            file.write(f"\nPercentage Sum Error: {pse}")  

    def predict(self, round_integer = True ,save_csv = True, save_log = True):
        # model prediction
        y_pred = self.best_model.predict(self.x_val)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        # if round prediction to integer
        if round_integer == True:
            y_pred = [round(num) for num in y_pred]

        # evaluate performance and save validation
        mae, rmse, r2, syt, syp, pse = self.evaluate_performance(self.y_val, y_pred)
        if save_csv == True:
            self.write_csv(y_pred)
        if save_log == True:
            self.write_results_log(mae, rmse, r2, syt, syp, pse)

if __name__ == "__main__":
    gs = Validation(
        modelfolder = r"PMF_Day_Breach_LR\\", 
        model_name_convention = "LR_PMF_Day_Breach_5CombinedIte"
    )
    
    gs.load_data(
        validation_data_csv = r"PMF_Day_lowBreach_Ite298_data.csv", 
        outputfolder = r"PMF_Day_Breach_LR\\",
        filter_criteria = None,
        fill_nodata = 0, 
        numerical_cols = ['PopU65', 'PopO65', 'TimeWarned', 'TimeMobilized', 'Structure_Number_of_Stories', 'Fording_Depth',
                          'Max_Depth', 'Max_Velocity', 'Max_DxV', 'Time_To_First_Wet'],
        categorical_cols = ['Warned','Mobilized', 'Structure_Stability_Criteria', 'Vehicle_Type'],
        idx_col = 'Index', 
        ref_col = 'Total_Life_Loss', 
        save_data = True,
        validation_name_convention = "PMF_Day_lowBreach_Ite298",        
    )
      
    gs.predict(round_integer = False, save_csv = True, save_log = True)
