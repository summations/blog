Title: Logistic Regression
Date: 2017-04-11 18:32
Category: tech
Tags: machinelearning, classification


To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values *y* we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and $y$ may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, $y\in{0,1}$. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given x(i), the corresponding $y^{(i)}$ is also called the label for the training example. 

### Hypothesis Representation

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict $y$ given $x$. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for $h_\theta(x)$ to take values larger than 1 or smaller than 0 when we know that $y ∈ {0, 1}$. To fix this, let’s change the form for our hypotheses $h_\theta(x)$ to satisfy $0≤h_\theta(x)≤1$. This is accomplished by plugging $\theta^Tx$ into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

<div class="mathsize">
\begin{align*}
h_\theta(x)=g(\theta^Tx)z=\theta^Txg(z)=11+e−z
\end{align*}
</div>


```python
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

#load the dataset
data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()
```


The following image shows us what the sigmoid function looks like:

<div class="figure">
<img alt="" src="/images/assets/logisticreg/sigmoidfunction.png" style="border:none; width:100%;">
</div>

The function $g(z)$, shown here, maps any real number to the $(0, 1)$ interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

$h_\theta(x)$ will give us the probability that our output is 1. For example, $h_\theta(x)$=0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

<div class="mathsize">
\begin{align*}
h_\theta(x)=P(y=1|x;\theta)=1−P(y=0|x;\theta)P(y=0|x;\theta)+P(y=1|x;\theta)=1
\end{align*}
</div>

### Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

<div class="mathsize">
\begin{align*}
h_\theta(x)≥0.5→y=1 \newline
h_\theta(x) < 0.5→y=0
\end{align*}
</div>

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5: that is when $g(z)≥0.5$ and when $z≥0$

Remember.

<div class="mathsize">
\begin{align*}
z=0, e_0=1&⇒g(z)=1/2 \newline
z→∞, e^{−∞}→0&⇒g(z)=1 \newline
z→−∞, e^{∞}→∞&⇒g(z)=0  
\end{align*}
</div>

So if our input to g is $\theta^TX$, then that means: $h_\theta(x)=g(\theta^Tx)≥0.5$ when $\theta^Tx≥0$.
From these statements we can now say: $\theta^Tx≥0⇒y=1$ and $\theta^Tx<0⇒y=0$.
The decision boundary is the line that separates the area where $y = 0$ and where $y = 1$. It is created by our hypothesis function.

Example:

<div class="mathsize">
\begin{align}
    \theta = 
    &\begin{bmatrix}
            5  \newline
            -1 \newline
            0  \newline
    \end{bmatrix} \newline
    y = 1 \quad &if \quad 5+(−1)x_1+0x_2≥0 \newline
    5−x_1 &≥ 0 \newline
    −x_1 &≥ −5 \newline
    x_1 &≤ 5 \newline
\end{align}
</div>

In this case, our decision boundary is a straight vertical line placed on the graph where x1=5, and everything to the left of that denotes $y$ = 1, while everything to the right denotes $y$ = 0.

Again, the input to the sigmoid function g(z) (e.g. $\theta^TX$) doesn't need to be linear, and could be a function that describes a circle (e.g. $z=\theta_0+\theta_1x^{2}_1+\theta_2x^{2}_2$) or any shape to fit our data.

### Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

<div class="mathsize">
\begin{align*}
    J(\theta) &= \frac{1}{m} \sum\limits_{i=1}^{m} Cost(h_\theta(x^{(i)}),y^{(i)}) \newline
    Cost(h_\theta(x),y) &= −log⁡(h_\theta(x)) \quad if \; y = 1 \newline
    Cost(h_\theta(x),y) &= −log⁡(1−h_\theta(x)) \quad if \; y = 0 \newline
\end{align*}
</div>

When y = 1, we get the following plot for $J(\theta)$ vs $h_\theta(x)$:
<div class="imgcap">
<img alt="/images/assets/logisticreg/costvshypo1.png" src="/images/assets/logisticreg/costvshypo1.png" style="border:none; width:30%;">
</div>

Similarly, when y = 0, we get the following plot for $J(\theta)$ vs $h_\theta(x)$:

<div class="imgcap">
<img alt="/images//assets/logisticreg/costvshypo2.png" src="/images//assets/logisticreg/costvshypo2.png" style="border:none; width:30%;">
</div>

Now our cost function becomes:

<div class="mathsize">
\begin{align*}
    Cost(h_\theta(x),y)&=0 \quad &if \; h_\theta(x)=y \newline
    Cost(h_\theta(x),y)&→∞ \quad &if \; y=0 \; and \; h_\theta(x)→1 \newline
    Cost(h_\theta(x),y)&→∞ \quad &if \; y=1 \; and \; h_\theta(x)→0 \newline
\end{align*}
</div>


If our correct answer $y$ is $0$, then the cost function will be $0$ if our hypothesis function also outputs $0$. If our hypothesis approaches $1$, then the cost function will approach infinity.

If our correct answer $y$ is $1$, then the cost function will be $0$ if our hypothesis function outputs $1$. If our hypothesis approaches $0$, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.
