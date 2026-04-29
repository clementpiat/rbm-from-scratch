## Hopfield Network

In **Hopfield Network**, we have binary neurons $x_i$, and $w_{i,j}$ weights which correspond to connections strengths between neurons. Then the energy we try to minimize is:

E(x) = $-\sum_{i,j} w_{i,j} x_i x_j$

&rarr; "Neurones that fire together, wire together", Hebb.

Weights define the **energy landscape**.
* Learning means updating weights.
* Inference is adjusting x to minimize the energy by descending along the surface (can be done by an iterative process).


## Restricted Boltzmann Machines

For RBM, we add hidden neurons, and the energy becomes: 

$
E(v,h)=-a^Tv-b^Th-v^TWh
$

&rarr; a, and b act as biases in case of binary vectors (Bernouilli variables).

We also add stochasticity in RBM. Neurons can be seen as particules with a probability to be in a given state / energy.

$
p(v,h)=\frac{1}{Z}e^{-\frac{E(v,h)}{T}}
$

&rarr; small T means deterministic like Hopfield Networks



## Free energy

If we define free energy as

$
p(v)=\frac{1}{Z}e^{-\frac{F(v)}{T}}
$

Applying the law of total probability we get

$
p(v)=\sum_h p(v,h)=\frac{1}{Z}\sum_he^{-\frac{E(v,h)}{T}}
$

which gives after applying log

$
F(v) = -T \log(\sum_he^{-\frac{E(v,h)}{T}})
$

and if we set T=1, and do the maths knowing that h is a binary vector, we get the following free will formula:

$
F(v) = -a^\top v - \sum_j \log\left(1 + e^{b_j + W_j^\top v}\right)
$
