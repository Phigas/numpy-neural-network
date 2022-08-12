import numpy as np

class Function:
    def __init__(self, run, derivate) -> None:
        self.run = run
        self.derivate = derivate

class Layer:
    def __init__(self, input_size, node_nr, activation) -> None:
        self.n_in = input_size[0] # input nodes
        self.m = input_size[1] # batch size
        self.n = node_nr
        
        # needs initialization
        self.weights = (np.random.rand(self.n, self.n_in)*0.5)-0.25
        self.b = np.zeros((self.n, self.m))

        self.activation = activation

    def forward_pass(self, X) -> np.ndarray:
        self.a_in = X
        self.Z = np.dot(self.weights, X) + self.b
        a = self.activation.run(self.Z)
        return a
    
    def backwards_pass(self, learning_rate, d_a):
        d_a = d_a.T
        d_z = d_a * self.activation.derivate(self.Z)
    
        new_d_a = np.dot(d_z.T, self.weights)
        
        d_w = np.dot(d_z, self.a_in.T)
        self.weights -= learning_rate * d_w
        
        d_b = d_z
        self.b -= learning_rate * d_b
        
        return new_d_a

class NeuralNetwork:
    def __init__(self, learning_rate, cost_function) -> None:
        self.layers = []
        self.input_shape = None
        self.output_shape = None
        
        self.learning_rate = learning_rate
        self.cost = cost_function
    
    def add_layer(self, layer):
        if self.layers == []:
            # first layer
            self.input_shape = (layer.n_in, layer.m)
            self.output_shape = (layer.n, layer.m)
            self.layers.append(layer)
        else:
            # if layer compatible
            if self.output_shape == (layer.n_in, layer.m):
                self.output_shape = (layer.n, layer.m)
                self.layers.append(layer)
            else:
                raise Exception('Layer does not match previous layer')

    def train_step(self, X, Y):
        Yhat = self.predict(X)
                
        loss = self.cost.run(Y, Yhat)
        
        d_a = self.cost.derivate(Y, Yhat)
        
        for layer in self.layers[::-1]:
            d_a = layer.backwards_pass(self.learning_rate, d_a)
            
        return loss
    
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

if __name__ == '__main__':
    # ==================================================
    # define functions
    # ==================================================
    
    sigma = lambda x: 1/(1+np.exp(-x))
    sigma_d = lambda x: np.exp(-x)/(1+np.exp(-x))**2
    sigmoid = Function(sigma, sigma_d)
    
    def cost(Y, Yhat):
        loss = []
        for y, yhat in zip(Y.T, Yhat.T):
            loss.append(-(y*np.log(yhat) + (1-y)*np.log(1-yhat)))
        return np.mean(loss)
    
    def cost_d(Y, Yhat):
        loss = []
        for y, yhat in zip(Y.T, Yhat.T):
            loss.append((yhat - y)/(yhat - yhat**2))
        loss = np.array(loss)
        return loss
    
    
    cost_function = Function(cost, cost_d)
    
    # ==================================================
    # build dataset
    # ==================================================
    
    import tensorflow_datasets as tfds
    import tensorflow as tf
    
    BATCH_SIZE = 4
    
    ds = tfds.load('mnist', split='train', shuffle_files=True)

    # Build your input pipeline
    ds = ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # ==================================================
    # define network
    # ==================================================
    
    nn = NeuralNetwork(.001, cost_function)
    
    nn.add_layer(Layer((28*28, BATCH_SIZE), 700, sigmoid))
    nn.add_layer(Layer((700, BATCH_SIZE), 400, sigmoid))
    nn.add_layer(Layer((400, BATCH_SIZE), 10, sigmoid))

    TRAIN_STEPS = 100
    
    costs = []
    i = 0
    
    for example in tfds.as_numpy(ds):
        print(f'=== running step: {i}')
        image, label = example["image"], example["label"]
        image = np.resize(image, (BATCH_SIZE, 28*28))
        image = np.moveaxis(image, [0,1], [1,0])
        label_vec = np.zeros((10, BATCH_SIZE))
        for u, l in enumerate(label):
            label_vec[l, u] = 1
        
        cos = nn.train_step(image, label_vec)
        print(f'=== cost of step was {cos}')
        costs.append(cos)
        
        i += 1
        if i == TRAIN_STEPS:
            break
        
print(costs) 
