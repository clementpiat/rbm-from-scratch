# Hopfield Network

In **Hopfield Network**, we have binary neurons $x_i$, and $w_{i,j}$ weights which correspond to connections strengths between neurons. Then the energy we try to minimize is:

E(x) = $-\sum_{i,j} w_{i,j} x_i x_j$

&rarr; "Neurones that fire together, wire together", Hebb.

Weights define the **energy landscape**.
* Learning means updating weights.
* Inference is adjusting x to minimize the energy by descending along the surface (can be done by an iterative process).


# (Restricted) Boltzmann Machines

RBM are neural networks defined by a joint probability inspired by statistical physics.

$$p(v,h)=\frac{1}{Z}e^{-\frac{E(v,h)}{T}}$$
$$E(v,h)=-a^Tv-b^Th-v^TWh$$

This energy is basically saying, neurons that often fires gives low energy, and neurons that often fire together gives low energy.

&rarr; for continuous inputs, we simply replace $a^Tv$ with a Gaussian term $\sum_i \frac{(v_i - \mu_i)^2}{2\sigma_i^2}$

&rarr; and everything is here: **the network is completely defined by the weights (W), the biases (a and b), and the temperature (T), and every equation that follows stems from the two above equations.**


## Free energy

We define the free energy of an input $F(v)$ such that

$p(v)=\frac{1}{Z}e^{-\frac{F(v)}{T}}$

Applying the law of total probability we get

$p(v)=\sum_h p(v,h)=\frac{1}{Z}\sum_he^{-\frac{E(v,h)}{T}}$

which gives after applying log

$F(v) = -T \log(\sum_he^{-\frac{E(v,h)}{T}})$

and if we do the maths, knowing that h is a binary vector, we get the following free energy formula:

$$F(v) = -a^\top v - T \sum_j \log\left(1 + e^\frac{b_j + W_j^\top v}{T}\right)$$

&rarr; We used the following intermediate result:
$\sum_h \exp\left(\sum_j h_j x_j\right) = \prod_j \left(1 + e^{x_j}\right)$

## Training objective

We want to maximize the likelihood of the training examples, that is, for a given example, minimize:

$L(v)=-\log(p(v))=-\log(\frac{1}{Z}\sum_{h}e^{-\frac{E(v,h)}{T}})=\log(Z)-\log(\sum_{h}e^{-\frac{E(v,h)}{T}})$

There are two opposing forces here, minimizing the free energy of the training example, while maximizing the energy within the partition function Z.

* $\log(Z)$ will be approximated with the Contrastive Divergence algorithm
* $-\log(\sum_{h}e^{-\frac{E(v,h)}{T}}) = \frac{F(v)}{T}$

If we compute the derivatives of the free energy, we get

* $\frac{\partial F}{\partial w_{i,j}} = - v_i h_j$
* $\frac{\partial F}{\partial a_{i}} = - v_i$
* $\frac{\partial F}{\partial b_{j}} = - h_j$
* $h_j=\sigma(x_j)$
* $x_j=\frac{b_j + \sum_{i} v_i W_{i,j}}{T}$

which gives when putting back into the loss:

$\frac{\partial L}{\partial w_{i,j}} = \frac{\partial \log(Z)}{\partial w_{i,j}} - \frac{v_i h_j}{T}$

And if we add a regularization term (weight decay):

$$\frac{\partial L}{\partial w_{i,j}} = \frac{\partial \log(Z)}{\partial w_{i,j}} - \frac{v_i h_j}{T} + \alpha w_{i,j}$$

## Sampling

$p(h_j=1|v)=\sigma(\frac{b_j + \sum_{i} v_i W_{i,j}}{T})$

And conversely:

$p(v_i=1|h)=\sigma(\frac{a_i + \sum_{j} h_j W_{i,j}}{T})$

These results are not straightforward, but they rely on the mutual independance of the $h_j$ given $v$. 

## ReLU Hidden units

The free energy becomes (at $T=1$):

$${F(v) = -a^\top v - \sum_j \frac{1}{2}\max(0, b_j + W_j^\top v)^2}$$


The derivatives become:

* $\frac{\partial F}{\partial w_{i,j}} = - v_i \max(0, x_j)$
* $\frac{\partial F}{\partial a_{i}} = - v_i$
* $\frac{\partial F}{\partial b_{j}} = - \max(0, x_j)$

And the sampling of h becomes:

$$h_j | v \sim \max(0, \mathcal{N}(x_j, \sigma(x_j))$$
