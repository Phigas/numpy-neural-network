# numpy-neural-network

As an excercise im building a simple neural network framework using only numpy.

The math is based on [this course](https://youtu.be/CS4cs9xVecg) by [Dr. Andrew Ng](https://www.andrewng.org).

## Notation

Variables

- $x^{(i)}\in\R^{n_x}$ is the ith input example
- $y^{(i)}\in\R$ is the ith true label
- $\hat{y}^{(i)}\in\R$ is the ith predicted label
- $b^{[i]}_j\in\R$ are the biases of the jth node of the ith layer
- $\omega^{[i]}_j\in\R^{1\times n_x}$ are the weights of jth node of the ith layer
- $n^{[i]}\in\R$ is the number of nodes of the ith layer

Dimension example of one layer computation:

- 4 nodes in layer and 3 input nodes to layer
- $z\in\R^{4\times 1}$, $W\in\R^{4\times 3}$, $x\in\R^{3\times 1}$, $b\in\R^{4\times 1}$
- $z = Wx+b$ no transpose because the $\omega$ vectors are in $W$ horizontally
- when vectorizing with batch size 5: $z\rightarrow Z\in\R^{4\times 5}$, $x\rightarrow X\in\R^{3\times 5}$, $b \rightarrow B\in\R^{4\times 5}$

Functions

- sigmoid function $\sigma(z)=\frac{1}{1+e^{-z}}$
- loss function $\mathcal{L}(\hat{y},y)=-(y\log(\hat{y})+(1-y)\log(1-\hat{y}))$
- cost function $\mathcal{J}(\sigma,b)=\frac{1}{m}\sum_{i=1}^m\mathcal{L}(\hat{y}^i,y^i)$
- logistic regression $\hat{y}=\sigma(z)$ with $z=\omega^Tx+b$
  - vectorized $\hat{Y}=\sigma(Z)$ with $Z=\omega^T\cdot X+b$
- gradient descent $\omega := \omega-\alpha\frac{\partial\mathcal{J}(\omega,b)}{\partial\omega}$
  - and $b := b-\alpha\frac{\partial\mathcal{J}(\omega,b)}{\partial b}$

Derivatives

- $\frac{\partial\mathcal{L}(\hat{y},y)}{\partial \hat{y}}=-\frac{y}{a}+\frac{1-y}{1-a}$
- $\frac{\partial\sigma(z)}{\partial z}=a-y$

Computing graph

- note: $\partial x=\frac{\partial\mathcal{J}}{\partial x}$
- isolate each mathematical step of the entire calculation as one substep
- For calculating $\partial x$ if $\mathcal{J}=y(x(...))$ we can use $\partial x=\frac{\partial\mathcal{J}}{\partial y}\frac{\partial y }{\partial x}$

## planned features

- [ ] layer base class
- [x] FC layer
- [ ] convolution layer
- [x] gradient descent
- [ ] momentum
- [ ] GAN or LSTM
