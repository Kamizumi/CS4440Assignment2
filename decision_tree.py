# -------------------------------------------------------------------------
# AUTHOR: Timothy Tsang
# FILENAME: decision_tree.py
# SPECIFICATION: Implementation of decision tree
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []
    accuracy_vals = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    # X =
    
    for row in data_training:
        try:
            encoded_taxable_income = float(row[2].lower().strip().replace('k', ''))
        except ValueError:
            encoded_taxable_income = 0

        is_single = 1 if row[1] == 'Single' else 0
        is_divorced = 1 if row[1] == 'Divorced' else 0
        is_married = 1 if row[1] == 'Married' else 0

        X.append([
            1 if row[0] == 'Yes' else 2,
            is_single,
            is_divorced,
            is_married,
            encoded_taxable_income
        ])
        
        
        

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for row in data_training:
        Y.append(1 if row[3] == 'Yes' else 2)
    

    
    #loop your training and test tasks 10 times here
    for i in range (10):
        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        #plt.show()

        #read the test data and add this data to data_test NumPy
        #--> add your Python code here
        # data_test =
        data_test_df = pd.read_csv('cheat_test.csv', sep =',', header = 0)
        data_test = np.array(data_test_df.values)[:,1:]

        correct_predictions = 0
        total_predictions = 0


        for data in data_test:
            
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            
            try:
                test_taxable_income = float(data[2].lower().strip().replace('k', ''))
            except ValueError:
                test_taxable_income = 0
            
            test_is_single = 1 if data[1] == 'Single' else 0
            test_is_divorced = 1 if data[1] == 'Divorced' else 0
            test_is_married = 1 if data[1] == 'Married' else 0

            test_X = [
                1 if data[0] == 'Yes' else 2,
                test_is_single,
                test_is_divorced,
                test_is_married,
                test_taxable_income
            ]

            class_predicted = clf.predict([test_X])[0]


            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            #--> add your Python code here
            true_label = 1 if data[3] == 'Yes' else 2

            if class_predicted == true_label:
                correct_predictions += 1

            total_predictions += 1

        curr_accuracy = correct_predictions / total_predictions
        accuracy_vals.append(curr_accuracy)
        




            #find the average accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
    average_accuracy = np.mean(accuracy_vals)

        #print the accuracy of this model during the 10 runs (training and test set).
        #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
        #--> add your Python code here
    print(f"Final accuracy when training on {ds}: {average_accuracy:.2f}")

