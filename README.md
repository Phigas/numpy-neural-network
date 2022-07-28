# numpy-neural-network

As an excercise im building a simple neural network framework using only numpy.

The math is based on [this course](https://youtu.be/CS4cs9xVecg) by [Dr. Andrew Ng](https://www.andrewng.org).

## Notation

Variables

- $x^{i}\in\R^{n_x}$ is one input example
- $X\in\R^{n_x\times m}$ is all input examples
- $y^{i}\in\R$ is one true label
- $Y\in\R^{1\times m}$ is all labels
- $\hat{y}^{i}\in\R$ is one predicted label
  - mathematically defined as $\hat{y}=P(y=1|x)$
- $\hat Y^{i}\in\R^{1\times m}$ are all predicted labels
- $b\in\R$
- $\omega\in\R^{1\times n_x}$

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
- [ ] FC layer
- [ ] convolution layer
- [ ] gradient descent
- [ ] momentum
- [ ] GAN or LSTM
