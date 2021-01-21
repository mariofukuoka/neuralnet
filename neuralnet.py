import numpy as np
import time

#TODO:
# numerical gradient checking
# early stopping?
# optimization algos

# neural network
class NeuralNet:
    def __init__(self):
        # init lists of per-layer variables
        self.L = 0
        self.func = []
        self.func_grad = []
        self.size = []
        self.w = []
        self.a = []
        self.z = []
        self.err = []
        self.grad = []
        
    def add_layer(self, num_units, activation='sigmoid'):
        # create all variables associated with the layer being added except
        # for the weight matrix, whose size depends on next layer as well
        self.L += 1
        self.size.append(num_units)
        self.func.append(activation_func_map[activation][0])
        self.func_grad.append(activation_func_map[activation][1])
        self.a.append(np.array([]))
        self.z.append(np.array([]))
        self.err.append(np.array([]))
        self.grad.append(np.array([]))
    
    def init(self):
        # iterate through pairs of neighboring layers and create weight matrices based on layer sizes
        for l in range(self.L-1):
            # init with random values around 0 to break symmetry
            self.w.append(np.random.randn(self.size[l+1], self.size[l]+1)*5)#np.sqrt(2/self.size[l]))
            
    def forward_pass(self, X):
        m = X.shape[0] # num of examples
        self.a[0] = X # initialize input
        for l in range(1, self.L):
            # add column of 1's for bias to prev layer before using it in the next step
            # (done at start of loop so that output layer skips this step)
            self.a[l-1] = np.c_[np.ones((m,1)), self.a[l-1]]
            # compute and store weighted input at layer i
            self.z[l] = self.a[l-1] @ self.w[l-1].T
            # compute and store activation at layer i
            self.a[l] = self.func[l](self.z[l])
            
    def predict(self, X):
        self.forward_pass(X)
        # return activation at last layer
        return self.a[self.L-1]

    def cost(self, X, y, reg_coeff=0):
        m = X.shape[0] # num of examples
        # mean square error plus regularization term
        class_predictions = self.predict(X)
        cost = np.sum(np.power(class_predictions - y, 2))/m \
                + reg_coeff*np.sum([np.sum(np.power(self.w[l], 2)) for l in range(self.L-1)])/(2*m)
        
        pred_labels = [np.argmax(classes) for classes in class_predictions]
        true_labels = [np.argmax(classes) for classes in y]
        hits = 0
        for i in range(len(pred_labels)):
            if pred_labels[i] == true_labels[i]:
                hits += 1
        acc = 100.0*hits/len(pred_labels)
        
        return cost, acc

    def backprop(self, X, y, reg_coeff=0):
        m = X.shape[0] # num of examples
        # compute and store error at last layer
        self.err[self.L-1] = self.a[self.L-1] - y
        # loop backwards over rest of layers (except for first one) and compute & store errors
        for l in range(self.L-2, 0, -1): 
            # use sub-matrix of weights without the bias column when multiplying
            self.err[l] = self.err[l+1] @ self.w[l][:,1:self.size[l]+1] \
                            * self.func_grad[l](self.z[l]) # m x l[l].size
        # compute and store gradients for each layer except last
        for l in range(self.L-1):
            self.grad[l] = (self.err[l+1].T @ self.a[l])/m \
                            + (reg_coeff*np.c_[np.zeros((self.size[l+1],1)), self.w[l][:,1:self.size[l]+1]])/m
    
    def update_weights(self, lr):
        # update weights by subtracting computed gradients multiplied by learning rate
        for l in range(self.L-1):
            self.w[l] -= lr*self.grad[l]
    
    def fit_batch(self, X_batch, y_batch, lr, reg_coeff):
        self.forward_pass(X_batch)
        self.backprop(X_batch, y_batch, reg_coeff)
        self.update_weights(lr)

    def fit(self, X, y, lr=0.1, epochs=1, batch_size=32, final_lr_fraction=1, reg_coeff=0):
        cost_history = []
        start_time = time.time()
        curr_lr = lr
        batch_start = 0
        for epoch in range(1, epochs+1):
            # propagate in batches
            batch_start = 0
            while (batch_end := batch_start + batch_size) < len(X):
                X_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                self.fit_batch(X_batch, y_batch, curr_lr, reg_coeff)
                batch_start = batch_end
            # propagate batch leftovers
            self.fit_batch(X[batch_start:-1], y[batch_start:-1], curr_lr, reg_coeff)

            # update lr based on num of epochs done
            curr_lr = lr*((epochs-(1-final_lr_fraction)*epoch)/epochs)

            # calculate and print stats
            epoch_cost, epoch_acc = self.cost(X, y)
            cost_history.append([epoch, epoch_cost, epoch_acc])
            curr_elapsed = time.time() - start_time
            print(f'epoch {epoch}, elapsed={curr_elapsed:0.1f}s, cost={epoch_cost:0.3f}, acc={epoch_acc:0.1f}, curr_lr={curr_lr:0.3f}')
        
        # calculate and print final stats
        total_elapsed = time.time() - start_time
        final_cost, final_acc = self.cost(X, y)
        print('===   finished fitting   ===')
        print(f'epochs={epoch}, total_elapsed={total_elapsed:0.2f}s' + '\n'\
              f'final_cost={final_cost:0.4f}, final_acc={final_acc:0.2f}' + '\n'\
              f'initial_lr={lr}, final_lr={curr_lr:0.4f}, reg_coeff={reg_coeff}')
        return cost_history

# activation funcs and their gradients

def sigmoid(z):
    # prevent overflow
    z = np.clip(z, -500, 500)
    
    return 1/(1 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return z * (z > 0)

def relu_grad(z):
    return z/z * (z > 0)

def softmax(z):
    # prevent overflow using property that softmax(x) = softmax(x - c)
    z = z - np.max(z, axis=-1, keepdims=True)

    out = np.exp(z)/np.reshape(np.sum(np.exp(z),1), (z.shape[0],1))
    return out

def softmax_grad(z):
    # never gets called if used only at output layer
    pass

activation_func_map = {
    'sigmoid':[sigmoid, sigmoid_grad],
    'relu':[relu, relu_grad],
    'softmax':[softmax, softmax_grad]
}