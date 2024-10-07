import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('future.no_silent_downcasting', True)

class LifeSim_Data:
    def __init__(self, roadSummary_csv, structureSummary_csv):
        self.df_structureSummary = pd.read_csv(structureSummary_csv)
        self.df_roadSummary = pd.read_csv(roadSummary_csv)

    def prepare_df(self, detailedOutput_csv):
        df_detailedOutput = pd.read_csv(detailedOutput_csv)
        df_detailedOutput_filtered = df_detailedOutput[[
            'Group_ID', 'Origin_Index', 'End_Road_Index',
            'PopU65', 'PopO65',
            'Warned', 'TimeWarned', 'Mobilized', 'TimeMobilized',
            'Vehicle_Type', 'Fording_Depth',
            'Unimpaired_Life_Loss','Impaired_Life_Loss',
            'Depth_Stability_Threshold', 'Velocity_Stability_Threshold', 'DV_Function_Threshold'
        ]]
        return df_detailedOutput_filtered      
    
    @staticmethod
    def process_summary(df, prefix):
        # set index as fid-1, this is to match the Origin_Index and End_Road_Index to the summary table
        df['Index'] = df['fid'] - 1

        # add prefix to distinguish road and structure data
        df = df.add_prefix(prefix)

        return df
    
    def merge_strcuture(self, df_detailedOutput):
        # structure groups are those are "Not Mobilized"
        df_structure = df_detailedOutput[df_detailedOutput['Mobilized'] == False]

        # road-stability variables does not apply for structure groups
        df_structure = df_structure.copy()
        df_structure['Vehicle_Type'] = np.NaN
        df_structure['Fording_Depth'] = np.NaN

        # merge data with summary table
        df_structureSummary = self.process_summary(self.df_structureSummary, 'Structure_')

        merged_df_structure = df_structure.merge(
            df_structureSummary[['Structure_Index', 'Structure_Max_Depth', 'Structure_Max_Velocity', 'Structure_Max_DxV', 'Structure_Time_To_First_Wet', 
                                 'Structure_Number_of_Stories', 'Structure_Stability_Criteria']],
            left_on='Origin_Index', right_on='Structure_Index', how='left')
        
        return merged_df_structure
    
    def merge_road(self, df_detailedOutput):
        # road groups are those are "Mobilized"
        df_road = df_detailedOutput[df_detailedOutput['Mobilized'] == True]
        
        # structure-stability variables does not apply for road groups
        df_road = df_road.copy()
        df_road['Structure_Number_of_Stories'] = np.NaN
        df_road['Structure_Stability_Criteria'] = np.NaN

        # merge data with summary table
        df_roadSummary = self.process_summary(self.df_roadSummary, 'Road_')

        merged_df_road = df_road.merge(
            df_roadSummary[['Road_Index', 'Road_Max_Depth_m', 'Road_Max_Velocity_m/s', 'Road_Max_DxV', 'Road_Time_To_First_Wet']],
            left_on='End_Road_Index', right_on='Road_Index', how='left')
        
        return merged_df_road
    
    def concat_road_structure(self, df_detailedOutput):
        # concat road and structure data
        merged_df_road = self.merge_road(df_detailedOutput)
        merged_df_structure = self.merge_strcuture(df_detailedOutput)

        df = pd.concat([merged_df_road, merged_df_structure], ignore_index=True)
        df.sort_values(by='Group_ID', inplace=True)

        # combine flood-related variables for structure and road into a single column
        df['Max_Depth'] = df['Structure_Max_Depth'].fillna(0) + df['Road_Max_Depth_m'].fillna(0)
        df['Max_Velocity'] = df['Structure_Max_Velocity'].fillna(0) + df['Road_Max_Velocity_m/s'].fillna(0)
        df['Max_DxV'] = df['Structure_Max_DxV'].fillna(0) + df['Road_Max_DxV'].fillna(0)
        df['Time_To_First_Wet'] = df['Structure_Time_To_First_Wet'].fillna(0) + df['Road_Time_To_First_Wet'].fillna(0)
        df['Time_To_First_Wet'] = df['Time_To_First_Wet'].apply(lambda x: 0 if x >3.4e+38 else x)

        # add structure and road life loss as total
        df['Total_Life_Loss'] = df['Impaired_Life_Loss'].fillna(0) + df['Unimpaired_Life_Loss'].fillna(0)

        # drop processed columns
        df = df.drop(columns = ['Road_Index', 'Structure_Index', 
                                'Structure_Max_Depth', 'Road_Max_Depth_m', 'Structure_Max_Velocity', 'Road_Max_Velocity_m/s', 
                                'Structure_Max_DxV', 'Road_Max_DxV', 'Structure_Time_To_First_Wet', 'Road_Time_To_First_Wet'])
        
        # for mobilization and warning,
        # if not mobilized/warned, the Time does not matter so apply NaN values
        df['Mobilized'] = df['Mobilized'].apply(lambda x: 1 if x == True else 0)
        df['TimeMobilized'] = df['TimeMobilized'].apply(lambda x: np.NaN if x <-3.4e+38 else x)
        df.loc[df['Mobilized'] == 0, 'TimeMobilized'] = np.NaN

        df['Warned'] = df['Warned'].apply(lambda x: 1 if x == True else 0)
        df['TimeWarned'] = df['TimeWarned'].apply(lambda x: np.NaN if x <-3.4e+38 else x)
        df.loc[df['Warned'] == 0, 'TimeWarned'] = np.NaN

        return df
    
    def concat_output(self, detailedOutput_csv_list, idx_col = 'Index'):
        self.detailedOutput_csv_list = detailedOutput_csv_list

        df_final = pd.DataFrame()
        # read all csv data in list
        for detailedOutput_csv in self.detailedOutput_csv_list:
            df_detailedOutput = self.prepare_df(detailedOutput_csv)
            df_processed = self.concat_road_structure(df_detailedOutput)
            df_final = pd.concat([df_final, df_processed], ignore_index=True)

        df_final[idx_col] = df_final.index
        return df_final

    def split_and_write(self, df, processed_csv, proportion_train_split = 0.8, random_state = 42):
        train_indices, test_indices = train_test_split(
            df.index, 
            train_size=proportion_train_split, 
            random_state=random_state, 
            shuffle=True
        )

        # filter the DataFrame to get train and test sets based on the indices
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]

        # construct file names based on the original CSV file name and save as CSV
        filename_convention = processed_csv.replace('.csv', '')
        train_df.to_csv(f'{filename_convention}_split_train.csv', index=False)
        test_df.to_csv(f'{filename_convention}_split_test.csv', index=False)
        print(f"Train data saved to: {filename_convention}_split_train.csv")
        print(f"Test data saved to: {filename_convention}_split_test.csv")

    def write_csv_w_split(self, df, processed_csv, split = False, proportion_train_split = None, random_state = None):
        # check if any data in a pandas DataFrame contains strings
        # if it does, it will cause error for regression unless the column is not used
        is_string = lambda x: isinstance(x, str)
        contains_strings = df.map(is_string)
        columns_with_strings = contains_strings.any()
        columns_with_strings_names = columns_with_strings[columns_with_strings].index.tolist()
        if len(columns_with_strings_names) >=1:
            print('Please replace following category column as numerical values (unless it is not used): ')
            print(columns_with_strings_names)
        else:
            pass

        # Save the final DataFrame
        processed_csv = processed_csv
        df.to_csv(processed_csv, index=False)
        print(f"Processed data saved to: {processed_csv}")

        if split == True:
            self.split_and_write(df, processed_csv, proportion_train_split, random_state)

    


