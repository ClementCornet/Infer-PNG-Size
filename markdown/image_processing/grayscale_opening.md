## Uncertainty

Entropy, or similar measures tend to estimate uncertainty or unpredictability of a signal. We work with images, signals than we actually are able to approximate - or predict. Then, evaluating the this approximation could give us a new idea about this unpredictability.

## Grayscale Opening

To do so, we use grayscale opening, succession of grayscale erosion and dilation. Erosion keeps the minimum value from the neighborhood of a point, while dilation keeps the maximum from a moving window. Then, the output of a whole grayscale opening process should be close from the input. Yet, differences should occur concerning some details, and some values that cannot be inferred from their neighborhood. In the end, we compute $\ell^0$, $\ell^1$, $\ell^2$,  and $H_S$ of the difference between the input and the output, to estimate the amount of correctly estimated pixels.