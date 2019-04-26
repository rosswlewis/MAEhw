
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    #print(transposedX)
    #print(np.dot(theta,transposedX))
    for interation in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate. 
        # Replace the following variables if needed
        #WITH VECTORS
        gradient = []
        for idx in range(len(theta)):
            gradient.append(sum((np.dot(theta,transposedX) - y)*X[:,idx])/m)
            #print(sum(np.dot(np.dot(theta,transposedX) - y,transposedX[idx])))
        #residualError =         
        
        #WITH INDEX
        #for idx, val in enumerate(theta):
            #runningSum = 0
            
            
            #sum(theta[0] + theta[1]*x1 + theta[2]*x2 - y)x1
            #theta
            #for index, row in enumerate(X):
                #print(row[1])
                #runningSum += (theta[0]*row[0] + theta[1]*row[1] + theta[2]*row[2] - y[index])*row[idx]
            #print(runningSum)
            #gradient.append(runningSum/m)
                
        #print(gradient)
        #gradient =  
        #theta = theta - alpha*gradient
        change = [alpha * x for x in gradient]
        #print('gradient',gradient)
        #print('theta', theta)
        #print('change',change)
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta; 
        #atmp = 0
        #for idx, row in enumerate(X):
        #    atmp += (y[idx] - (theta[0]*row[0] + theta[1]*row[1] + theta[2]*row[2]))**2
        #atmp = atmp/m
        #print(atmp)
        #print((y - np.dot(theta,transposedX))**2)
        atmp = (sum((np.dot(theta,transposedX) - y)**2)/m)
        arrCost.append(atmp)
        # cost = (1 / m) * np.sum(residualError ** 2)
        ################PLACEHOLDER4 #start##########################

    return theta, arrCost
