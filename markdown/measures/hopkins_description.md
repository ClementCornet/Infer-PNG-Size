

## Description

 It behaves like a statistical hypothesis test, where the null hypothesis is that the points are uniformly distributed. To compute it on a set $X$ of $n$ points in $d$ dimensions:


- Generate $\tilde{X}$, a random sample of $m \ll n$ datapoints from $X$ (i.e. 5%)
- Generate $Y$, a set of $m$ randomly and uniformly distributed datapoints
- Note $u_i$ the minimum distance of $y_i \in Y$ to its nearest neighbor in $X$
- Note $w_i$ the minimum distance of $\tilde{x_i} \in \tilde{X}$ to its nearest neighbor in $X$

Then compute $H$ the Hopkins Statistic defined by: 

$$
    H = \frac{\sum_{i=1}^{m} u_i^d}{\sum_{i=1}^{m} u_i^d + \sum_{i=1}^{m} w_i^d}
$$

$H$ is bounded between 0 and 1. A value close to 1 indicates that the data has a high clustering tendency, its data points are typically much closer to other data points than to randomly generated ones. A value close to 0 indicates uniformly spaced data, and values around 0.5 indicate uniform data. Note that uniform datasets with a few outliers could obtain a high $H$.