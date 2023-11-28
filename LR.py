import pandas as pd
import numpy as np

def prepare_data(path):
  
    target_val_col = []
    features_col = []
    running = True
    
    # reading in data
    data = pd.read_csv(path)
    print("Please choose features and target values from this list: "+ str(list(data))+".\n")
    target_val_col.append(str(input("Specify one target value:")))
    while running:
        features_col.append(str(input("Specify features:")))
        ui = str(input("Do you want to add another feature?(Y/N)"))
        if ui=="N":
            print("Exiting...")
            running = False
        else:
            pass
    print("Target value:", target_val_col)
    print("Features: ", features_col)
    
    # preparing data for features and target value
    target_value = data[target_val_col[0]]
    features = data[features_col]

    return target_value, features

def cost(weights, y, f):
    # make sure indexes pair with number of rows
    weights.reset_index()
    f.reset_index()

    # Extract the first row 
    first_val_weights = weights.iloc[:1]

    # exclude first row
    weights_excluded_first_row = weights.iloc[1:]
    
    # Multiply values of the weights DataFrame with every row of the features DataFrame
    result_sum_by_row = (weights_excluded_first_row.values * f.values).sum(axis=1)

    # Add the first weight we excluded
    result_sum_by_row += first_val_weights.values.flatten()

    result_df = pd.DataFrame(result_sum_by_row, columns=['Result'])

    return result_df

def linear_regression(target, features):
    # initialize weights
    # +1 to represent the single weight with no feature
    weights = pd.DataFrame([1 for _ in range(len(features)+1)])
    print(cost(weights, target, features))

def main():
    target, features = prepare_data("Housing.csv")
    linear_regression(target, features)
    

if __name__ == "__main__":
    main()