Using most metrics described in _Comparing measures of sparsity
N Hurley, S Rickard - IEEE Transactions on Information Theory, 2009_

- $\ell^0 = \# \{ j, c_j = 0 \}$
- $\ell^0_\epsilon = \# \{ j, c_j \leq \epsilon \}$
- $\ell^1 = \sum_j c_j$
- $\ell^p = (\sum_j c_j^p)^{1/p}$
- $tanh_{a,b} = \sum_j tanh((ac_j)^b)$
- $log = \sum_j log(1+c_j^2)$
- $\kappa_4 = \frac{\sum_j c_j^4}{(\sum_j c_j^2)^2}$
- $\ell^p_- = \sum_j c_j^p$ with  $p<0$
- $H_G = -\sum_j log(c_j^2)$
- $H_S = -\sum_j \frac{c_j^2}{||\vec{c}||^2_2} log(\frac{c_j^2}{||\vec{c}||^2_2})^2$
- $H_{S'} = -\sum_j c_j \times log(c_j)^2$
- $Hoyer = \frac{\sqrt{N} - \frac{\ell^1}{\ell^2}}{\sqrt{N}-1}$
- $Gini = \frac{2 \sum_j j\times c_j}{n \sum_j c_j} - \frac{n+1}{n}$