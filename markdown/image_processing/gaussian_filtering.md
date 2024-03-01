## Details

Clustering tendency (Hopkins Statistic), unpredictability (Shannon Entropy) or even stationnarity ($\ell^0$ norm over Paeth filtered images) : most of our indicators seem linked to the notion of details. Then, we could use image processing techniques to extract details from an image, and compute known measures over the result.

## Difference of Gaussians

To do so, we compute the difference between 2 Gaussian filtered images with different orders (0.5 and 1). In the end, we compute the $\ell^0$ norm of the result, that can be thought as a "level of details" indicator.

$$
DoG (I) = G^{(1/2)}(I) - G^{(1)}(I)
$$