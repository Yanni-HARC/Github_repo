# :rocket: A Machine Learning Approach to Identifying Key Factors Influencing PLL Estimates

## Overview
This repository contains the implementation of machine learning models and SHAP analysis presented at the 2024 ANCOLD Conference in the paper titled **"A Machine Learning Approach to Identifying Key Factors Influencing PLL Estimates"** by Yanni Wang, Albert Shen, David Stephens, and Ze Jiang.

The study focuses on advancing our understanding of Potential Loss of Life (PLL) estimates by employing **Explainable Artificial Intelligence (XAI)** techniques. These methods are applied to explore and interpret the complex relationships that influence PLL estimates in dynamic simulation models such as LifeSim. By leveraging XAI, we provide actionable insights into key factors driving PLL estimates, helping to improve decision-making and resource allocation in dam safety assessments.

## Features
:rocket: **Machine Learning Model**
Explore how machine learning models, specifically gradient boosting, can improve the understanding of complex relationships in LifeSim simulation data.

:robot: **Explainable AI (XAI)** 
Utilize SHapley Additive exPlanations (SHAP) to improve our current system understanding and tramsform the black-box nature into human-interpretable formats.

:bar_chart: **Data-Driven Insights** 
Identify key factors influencing Potential Loss of Life (PLL) estimates, supporting data-driven decision-making in dam safety assessments.


## Getting Started
###  Prerequisites
Before you begin, make sure you have the following installed:
- :snake: Python 3.11+
- :package: Necessary dependencies listed in requirements.txt
    - `numpy` for numerical operations
    - `pandas` for data processing
    - `sklearn` for machine learning model development
    - `shap` for model interpretation
    - `matplotlib` for figure plotting

### Installation
Clone the repository:

```bash
git clone https://github.com/Yanni-HARC/InterpretableML_wLifeSim.git
cd InterpretableML_wLifeSim
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### 1. :wrench: Extract LifeSim Simulations and Save as CSV

To begin, extract individual LifeSim simulation results, including:
- Detailed_Output Results
- Road Summary
- Structure Summary

Data preprocessing can be done using preprocess.py. Here's how to run it:

```python
# load data
data = LifeSim_Data(
    roadSummary_csv = road_summary_csv_filepath, 
    structureSummary_csv = structure_summary_csv_filepath
    )

detailedOutput_csv_list = [
    detailed_output_results_csv1_filepath, detailed_output_results_csv2_filepath, 
    ...
    ]

df = data.concat_output(detailedOutput_csv_list)

# replace string in structure stability and vehicle type to int
structure_stability_dict = {
        'Masonry': 1,
        'Wood-Anchored': 2,
        'Manufactured': 3
    }
vehicle_type_dict = {
        'High Clearance': 1,
        'Low Clearance': 2
    }    

df['Structure_Stability_Criteria'] = df['Structure_Stability_Criteria'].replace(structure_stability_dict)
df['Vehicle_Type'] = df['Vehicle_Type'].replace(vehicle_type_dict)

# write csv
data.write_csv_w_split(df, output_csv, split = True)
```

### 2. :brain: Run the Machine Learning Model

Next, train and test your machine learning model using train_and_test.py as follows:

```python
# choose the model and output
model = ML_Algorithm(
        model = 'Linear', # 'HistGradientBoosting_Regressor' or 'Linear'
        outputfolder = output_folderpath
        )

# define numerical and categorical columns
numerical_cols = ['PopU65', 'PopO65', 'TimeWarned', 'TimeMobilized', 'Structure_Number_of_Stories', 'Fording_Depth',
                          'Max_Depth', 'Max_Velocity', 'Max_DxV', 'Time_To_First_Wet']
categorical_cols = ['Warned','Mobilized', 'Structure_Stability_Criteria', 'Vehicle_Type'] 

# load training and testing data for the model
model.load_data(
    train_data_csv = train_data_csv_filepath, 
    test_data_csv = test_data_csv_filepath, 
    filter_criteria = None, # "structure" or "road" or None
    fill_nodata = None, # fill no data with a value or None
    numerical_cols = numerical_cols,
    categorical_cols = categorical_cols,
    idx_col = 'Index', 
    ref_col = 'Total_Life_Loss', 
    save_data = True,
    name_convention = 'test'
    )     

# model training
model.perform_grid_search(
    kf_config = None, 
    save_best_model = True,
    save_cv_results = True
    )

# model testing
model.predict(
    round_integer = False, 
    save_csv = True, 
    save_log = True
    )
```
### 3. :repeat: Cross-Validation for Robustness Testing

Run cross-validation using robustness.py to evaluate the robustness of the model across various scenarios:

```python
model = Validation(
    modelfolder = model_folderpath, 
    model_name_convention = 'test'
    )
    
model.load_data(
    validation_data_csv = validation_data_csv_filepath, 
    outputfolder = output_folderpath,
    filter_criteria = None,
    fill_nodata = 0, 
    numerical_cols = numerical_cols,
    categorical_cols = categorical_cols,
    idx_col = 'Index', 
    ref_col = 'Total_Life_Loss', 
    save_data = True,
    validation_name_convention = 'test',        
    )
      
model.predict(round_integer = False, save_csv = True, save_log = True)
```
### 4. :bar_chart: Visualize Learning Curves

To visualize the learning curve of your model, refer to plot_learning_curve.ipynb. This notebook allows you to track model performance over time and assess overfitting or underfitting.

###  5. :mag_right: SHAP Analysis

To visualize and interpret the model’s predictions using SHAP, see plot_shap_analysis.ipynb. This notebook generates SHAP summary and dependence plots, providing insights into the contribution of each variable to the model's predictions.

## License
This project is licensed under the MIT License.


## Contact
For any questions or feedback, feel free to reach out to:
Yanni Wang - yanni.wang@harc.com.au
