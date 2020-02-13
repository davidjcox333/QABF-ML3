import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from numpy.random import seed
from keras.regularizers import l1
import random



seed = 6
np.random.seed(seed)

dataframe = pandas.read_csv("FullTest.csv", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,0:8].astype(float)
y_orig = dataset[:,8:12].astype(float)



#Function 1: All functions may change in modification 
def generate_data1(training_data, number_samples):
    generated_set = np.empty((0,12))
    sampling_set1= [1,-1,0]
    sampling_set2= [1,2,3] 
    
    for i in range(number_samples):
        row_index = i % len(training_data)
        new_row = training_data[row_index,:].copy().reshape(1,12)
        for j in range(4):
            change_endors = np.random.choice(sampling_set1,1)
            change_severity= np.random.choice(sampling_set2, 1)*change_endors
            new_row[0,j] = new_row[0,j] + change_endors
            if new_row[0, j] > 5:
                new_row[0,j] = 5
            if new_row[0,j] <= 0:
                new_row[0,j+4] = 0
            else: 
                new_row[0,j+4] = new_row[0,j+4] + change_severity
                
            if new_row[0,j+4] > new_row[0,j]*3 :
                new_row[0, j+4] = (new_row[0,j].copy())*3
                
            if new_row[0,j+4] < new_row[0,j] :
                new_row[0, j+4] = (new_row[0,j].copy())
            
        new_row[new_row < 0] = 0

        generated_set = np.vstack((generated_set, new_row.flatten()))
    
    return generated_set

    # Function 2: The values of a single function is modified
def generate_data2(training_data, number_samples):
    random.seed(6)
    generated_set = np.empty((0,12))
    sampling_set1= [1,-1]
    sampling_set2= [1,2,3] 
    
    for i in range(number_samples):
        row_index = i % len(training_data)
        new_row = training_data[row_index,:].copy().reshape(1,12)
        j = random.randint(0,3)
        change_endors = np.random.choice(sampling_set1,1)
        change_severity= np.random.choice(sampling_set2, 1)*change_endors
        new_row[0,j] = new_row[0,j] + change_endors
        if new_row[0, j] > 5:
            new_row[0,j] = 5
        
        if new_row[0,j] <= 0:
            new_row[0,j+4] = 0
        else: 
            new_row[0,j+4] = new_row[0,j+4] + change_severity
        
        if new_row[0,j+4] > new_row[0,j]*3 :
            new_row[0, j+4] = (new_row[0,j].copy())*3
        
        if new_row[0,j+4] < new_row[0,j] :
            new_row[0, j+4] = (new_row[0,j].copy())
            
        
        new_row[new_row < 0] = 0
        
        
        generated_set = np.vstack((generated_set, new_row.flatten()))
    
    return generated_set

    #Function 3 : The value of all functions is changed (excluding excluding the values of the actual function)
def generate_data3(training_data, number_samples):
    
    generated_set = np.empty((0,12))
    sampling_set1= [0,1,-1]
    sampling_set2= [1,2,3] 
    
    for i in range(number_samples):
        row_index = i % len(training_data)
        new_row = training_data[row_index,:].copy().reshape(1,12)
        nonfunction_index = list((np.where(new_row[0,8:12] == 0))[0])
        for j in nonfunction_index:
            change_endors = np.random.choice(sampling_set1,1)
            change_severity= np.random.choice(sampling_set2, 1)*change_endors
            new_row[0,j] = new_row[0,j]  + change_endors
            if new_row[0, j] > 5:
                new_row[0,j] = 5
            if new_row[0,j] <= 0:
                new_row[0,j+4] = 0
            else: 
                new_row[0,j+4] = new_row[0,j+4] + change_severity
            
            if new_row[0,j+4] > new_row[0,j]*3 :
                new_row[0, j+4] = (new_row[0,j].copy())*3
                                
            if new_row[0,j+4] < new_row[0,j] :
                new_row[0, j+4] = (new_row[0,j].copy())
            
        new_row[new_row < 0] = 0
        
        
        generated_set = np.vstack((generated_set, new_row.flatten()))
    return generated_set
        
# Function 4 : The value of a single function is changed (excluding the values of the actual function)
def generate_data4(training_data, number_samples):
    
    generated_set = np.empty((0,12))
    sampling_set1= [1,-1]
    sampling_set2= [1,2,3] 
    
    for i in range(number_samples):
        row_index = i % len(training_data)
        new_row = training_data[row_index,:].copy().reshape(1,12)
        
        nonfunction_index = list((np.where(new_row[0,8:12] == 0))[0])

        j = np.random.choice(nonfunction_index, 1)
        
        change_endors = np.random.choice(sampling_set1,1)
        change_severity= np.random.choice(sampling_set2, 1)*change_endors
        new_row[0,j] = new_row[0,j]+ change_endors
        if new_row[0,j] > 5:
            new_row[0,j] = 5
        
        if new_row[0,j] <= 0:
            new_row[0,j+4] = 0
        else: 
            new_row[0,j+4] = new_row[0,j+4] + change_severity

        if new_row[0,j+4] > new_row[0,j]*3 :
            new_row[0, j+4] = (new_row[0,j].copy())*3
        
        if new_row[0,j+4] < new_row[0,j] :
            new_row[0, j+4] = (new_row[0,j].copy())

        new_row[new_row < 0] = 0

        
        generated_set = np.vstack((generated_set, new_row.flatten()))
    
    return generated_set
################################################################################################################

model = Sequential()
model.add(Dense(16, activation='linear', input_dim=X_orig.shape[1]))
model.add(Activation('relu'))
model.add(Dense(y_orig.shape[1], activation='sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



i = 0
j = 0
for p in range(0, 49):
	X_copy = X_orig[(p):(p+1)]  #Slice the ith element from the numpy array
	y_copy = y_orig[(p):(p+1)]
	X_model = X_orig
	y_model = y_orig  #Set X and y equal to samples and labels


	X_model = np.delete(X_model, p, axis = 0)  #Create a new array to train the model with slicing out the ith item for LOOCV
	y_model = np.delete(y_model, p, axis = 0)

	train_set = np.concatenate((X_model, y_model), axis = 1) #combine numpy matrices 

			
	#generated_data1 = generate_data1(train_set, 1000)  #run sample generator function
	generated_data2 = generate_data2(train_set, 1000)
	#generated_data3 = generate_data3(train_set, 1000)
	#generated_data4 = generate_data4(train_set, 1000)

	X_train = generated_data2[:,0:8].astype(float)
	y_train = generated_data2[:,8:12].astype(float)

	X_train2 = np.concatenate([X_train, X_model])
	y_train2 = np.concatenate([y_train, y_model])



	model.fit(X_train2, y_train2, epochs=40, batch_size=15, verbose=0)
	prediction = (model.predict(X_copy))
	prediction[prediction>=0.5] = 1
	prediction[prediction<0.5] = 0
	print(prediction, y_copy)
	if np.array_equal(y_copy, prediction):
		j = j + 1
		#print(y_copy, prediction)
	if np.not_equal:
		#print(y_copy, prediction)
		pass
print(j/49)

