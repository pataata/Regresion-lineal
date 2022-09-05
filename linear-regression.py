#Ruben Sanchez Mayen A01378379

import random

def h(samples, params):
    y = 0
    for i in range(len(params)):
        y = y + params[i]*samples[i]
    return y

def show_error(params, samples,y):
    error_acum =0
    for i in range(len(samples)):
        hyp = h(params,samples[i])
        #print( "hyp  %f  y %f " % (hyp,  y[i]))
        error=hyp-y[i]
        error_acum=+error**2 
    mean_error_param=error_acum/len(samples)
    print('Average square error:',mean_error_param)

def GD(params, samples, y, alfa):  
    temp = list(params)
    for j in range(len(params)):
        acum =0
        for i in range(len(samples)):
            error = h(params,samples[i]) - y[i]
            acum = acum + error*samples[i][j]  
        temp[j] = params[j] - alfa*(1/len(samples))*acum  
    return temp

def normalization(samples):
    acum = 0
    for i in range(len(samples)):
        max = samples[i][0]
        for j in range(len(samples[i])):
            acum += samples[i][j]
            if(samples[i][j] > max):
                max = samples[i][j]
        avg = acum/len(samples)
        for j in range(len(samples[i])):
            samples[i][j] =  (samples[i][j] - avg)/max
    return samples

def generate_samples(m):
    #y = 2x1 + 3x2 + 4.5x3 + 10
    x_samples = []
    y_samples = []
    for i in range(m):
        y = i*2 + (i-1)*3 + (i-2)*4.5 + 10
        y += random.randint(-2,2) #Add noise
        x_samples.append([i,i-1,i-2])
        y_samples.append(y)
        

    return [x_samples,y_samples,]

# Hyper parameters
m = 100
alpha = .0001
epochs = 0

#  Data preparation
samples_xy = generate_samples(m)
params = [0,0,0,0]
samples = samples_xy[0]
y = samples_xy[1]
prediction = [1, 4, 2, 8]

# Add beta0 to samples
for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i]=  [1]+samples[i]
	else:
		samples[i]=  [1,samples[i]]

#Multiple tests changing the train and test data sets
for i in range(5):
    print('\nIteration: ',i+1)
    print()
    x_train = []; y_train = []; x_test = []; y_test = []
    
    for j in range(len(samples)): # fill the train/test data sets
        n = random.randint(0,m-1)
        if(j <= len(samples)*.8 ):
            #train data set
            x_train.append(samples[n])
            y_train.append(y[n])
        else:
            #test data set
            x_test.append(samples[n])
            y_test.append(y[n])

    #print('Train data set: ',x_train,y_train)
    #print('Test data set: ',x_test,y_test)

    epochs = 0
    while True:  #  run gradient descent until local minima is reached
        past_params = list(params)
        params=GD(params, x_train,y_train,alpha)	
        epochs = epochs + 1
        if(past_params == params or epochs == 1000):   #  local minima is found when there is no further improvement
            #print ("samples:")
            #print(samples)
            print ("final params:",params)
            show_error(params, x_test, y_test)
            print ('Estimated Y for',prediction,':',end=' ')
            estimated_y = 0
            for j in range(len(params)):
                estimated_y += params[j] * prediction[j]
            print(estimated_y)
            break




