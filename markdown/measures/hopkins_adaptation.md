## Adaptation to our purpose

$H$ is usually used in a clustering framework. Yet, it requires coordinates in $d$ dimensions. In our sparsity measurement study, we build a coordinate map, using index in our data stream/image as coordinate.
$$
0101 \rightarrow (0,0),(1,1),(2,0),(3,1)
$$
We then use standard scaling ($z = \frac{x-\mu}{\sigma}$) to avoid giving too much importance to the index. Finally, we compute the Hopkins Statistic on the obtained $d+1$ dimensions result. For RGB images, we compute the Hopkins statistic over all 3 channels, and return the average.