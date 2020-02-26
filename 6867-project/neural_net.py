import numpy as np
from numpy import *

def reLu(x):
    return maximum(x,0)
def reLu_p(x):
    return (sign(x)+1)/2
def softmax(z):
    return exp(z)/(sum(exp(z)))
def softmax_p(z,y):
    return softmax(z)-y
def train(weights,biases,func,func_p,output_difference,x_val,y_val,eta_val):
    z_val =evaluate_neural_nets(weights,biases,func,x_val)
    assert len(z_val)==len(weights)+1
    assert len(weights)==len(biases)
    delta_val=None
    ###do back propagation
    for ele in range(len(weights),0,-1):
        if delta_val is None:
            delta_val= softmax_p(z_val[ele][0],y_val)
        new_delta_val=multiply(weights[ele-1].T.dot(delta_val),reLu_p(z_val[ele-1][0]))
        biases[ele-1]-= eta_val*delta_val
        weights[ele-1]-=eta_val*delta_val.dot(z_val[ele-1][1].T)
        delta_val=new_delta_val
    return weights,biases
def evaluate_neural_nets(weights,biases,func,x_val):
    assert len(weights)==len(biases)
    list_of_activates=[(x_val,x_val)]
    for ele in range(len(weights)):
        z_val=weights[ele].dot(list_of_activates[-1][1])+ biases[ele]
        list_of_activates.append((z_val,(func if ele < len(weights)-1 else softmax)(z_val) ) )
    return list_of_activates
def cross_eval_loss(output,y_val):
    assert shape(output)==shape(y_val)
    return -np.sum(multiply(y_val,log(output)))
def error(weights,biases,func,output,X,Y):
    return sum(cross_eval_loss(evaluate_neural_nets(weights,biases,func,mat(X[ele]).T)[-1][-1],mat(Y[ele]).T) for ele in range(len(X)))/len(X)

def weight_maker(ele1,ele2):
    return random.normal(0,ele1**-0.5,(ele1,ele2))

def train_and_val_neural_net(size_of_layers,learning,trainX,trainY,validateX,validateY):
    size_of_window=20
    window=[None for ele in range(size_of_window)]
    max_epochs=1000
    weights,biases=[],[]
    for ele in range(1,len(size_of_layers)):
        weights.append(weight_maker(size_of_layers[ele],size_of_layers[ele-1]))
        biases.append(zeros((size_of_layers[ele],1)))
    old_weights,old_biases=None,None
    for ele1 in range(max_epochs):
        for ele2 in range(len(trainX)):
            x_val,y_val=mat(trainX[ele2]).T,mat(trainY[ele2]).T
            train(weights,biases,reLu,reLu_p,softmax,x_val,y_val,learning)
            if False and ele2 % 40 == 0:
                validateError=error(weights,biases,reLu,softmax,validateX,validateY)
                if all(window) and validateError >=sum(window)/size_of_window:
                    print("HEY",ele1)
                    return old_weights,old_biases
                else:
                    window = [validateError] + window[:-1]
                    old_weights,old_biases=weights[:],biases[:]
    return weights[:],biases[:]
def extract_XY_path(path_to_use):
    data=loadtxt(path_to_use)
    X=data[:,0:2]
    Y=data[:,2:3]
    Y=array([eye(1,2, 1 if y_ele ==1 else 0) for y_ele in Y])
    return X,Y
def best_learn(trainX,trainY,validateX,validateY,testX,testY,shape_of_layer):
    print(shape_of_layer)
    best_value=1000
    for learn_rate in [0.1, 0.01 , 0.001 , 0.0001 , 0.00001 , 0.000001]: #can change the learning rates here
        weights,biases = train_and_val_neural_net(shape_of_layer,learn_rate,trainX,trainY,validateX,validateY)
        value_error=error(weights,biases,reLu,softmax,validateX,validateY)
        print("Error for learning rate %f is %f" %(learn_rate,value_error))
        if value_error < best_value:
            best_value,best_learn,best_w=value_error,learn_rate,(weights,biases)
    weights,biases = best_w
    print(best_learn)
    print('Training error', error(weights,biases,reLu,softmax,trainX,trainY))
    print('Validation error',error(weights,biases,reLu,softmax,validateX,validateY))
    print('Test error',error(weights,biases,reLu,softmax,testX,testY))
    
if __name__ == '__main__':
    X=[]
    Y=[]
    for ele in range(10):
        X.append(2/255.*loadtxt('' % ele)-1) #include path in here
        Y.append([eye(1,10,ele) for x_ele in X[-1]])
    trainX,trainY=[concatenate([d_ele[:20] for d_ele in XY]) for XY in [X, Y]]
    validateX,validateY=[concatenate([d_ele[20:35] for d_ele in XY]) for XY in [X, Y]]
    testX,testY = [concatenate([d_ele[35:50] for d_ele in XY]) for XY in [X, Y]]
    for shape_of_layer in [[784,2,10],[784,20,10],[784,2,2,10],[784,20,20,10]]:
        best_learn(trainX,trainY,validateX,validateY,testX,testY,shape_of_layer)
    
        
        
