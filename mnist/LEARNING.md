# Hopfield Network

In **Hopfield Network**, we have binary neurons $x_i$, and $w_{i,j}$ weights which correspond to connections strengths between neurons. Then the energy we try to minimize is:

E(x) = $-\sum_{i,j} w_{i,j} x_i x_j$

&rarr; "Neurones that fire together, wire together", Hebb.

Weights define the **energy landscape**.
* Learning means updating weights.
* Inference is adjusting x to minimize the energy by descending along the surface (can be done by an iterative process).


# Restricted Boltzmann Machines

For RBM, we add hidden neurons, and the energy becomes: 

$$E(v,h)=-a^Tv-b^Th-v^TWh$$



We also add stochasticity in RBM. Neurons can be seen as particules with a probability to be in a given state / energy.

$$p(v,h)=\frac{1}{Z}e^{-\frac{E(v,h)}{T}}$$

&rarr; small temperature T correspond to a deterministic system like Hopfield Networks

&rarr; everything is here: **the network is completely defined by the weights (W), the biases (a and b), and the temperature (T), and every equation that follows stems from the two above equations.**


## Free energy

If we define free energy as

$p(v)=\frac{1}{Z}e^{-\frac{F(v)}{T}}$

Applying the law of total probability we get

$p(v)=\sum_h p(v,h)=\frac{1}{Z}\sum_he^{-\frac{E(v,h)}{T}}$

which gives after applying log

$F(v) = -T \log(\sum_he^{-\frac{E(v,h)}{T}})$

and if we do the maths knowing that h is a binary vector, we get the following free energy formula:

$$F(v) = -a^\top v - T \sum_j \log\left(1 + e^\frac{b_j + W_j^\top v}{T}\right)$$

&rarr; We used the following intermediate result:
$\sum_h \exp\left(\sum_j h_j x_j\right) = \prod_j \left(1 + e^{x_j}\right)$

## Training objective

We want to maximize the likelihood of the training examples, that is, for a given example, minimize:

$L(v)=-\log(p(v))=-\log(\frac{1}{Z}\sum_{h}e^{-\frac{E(v,h)}{T}})=\log(Z)-\log(\sum_{h}e^{-\frac{E(v,h)}{T}})$

There are two opposing forces here, minimizing the free energy of the training example, while maximizing the partition function Z (which could be seen as the total energy of the system).

* $\log(Z)$ will be approximated with the Contrastive Divergence strategy
* $-\log(\sum_{h}e^{-\frac{E(v,h)}{T}}) = \frac{F(v)}{T}$

If we compute the derivative of the free energy, we get

$\begin{cases}
\frac{\partial F}{\partial w_{i,j}} = - v_i h_j \\
\frac{\partial F}{\partial a_{i}} = - v_i \\
\frac{\partial F}{\partial b_{j}} = - h_j \\
h_j=\sigma(x_j) \\
x_j=\frac{b_j + \sum_{i} v_i W_{i,j}}{T}
\end{cases}$

which gives when putting into the loss:

$\frac{\partial L}{\partial w_{i,j}} = \frac{\partial \log(Z)}{\partial w_{i,j}} - \frac{v_i h_j}{T}$

And if we add a regularization term (weight decay):

$$\frac{\partial L}{\partial w_{i,j}} = \frac{\partial \log(Z)}{\partial w_{i,j}} - \frac{v_i h_j}{T} + \alpha w_{i,j}$$

## Sampling

$p(h_j=1|v)=\sigma(\frac{b_j + \sum_{i} v_i W_{i,j}}{T})$

And conversely:

$p(v_i=1|h)=\sigma(\frac{a_i + \sum_{j} h_j W_{i,j}}{T})$

These results are not straightforward, but they rely on the mutual independance of the $h_j$ given $v$. 
