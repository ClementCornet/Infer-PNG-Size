## Cardinality
We've seen that images with very few different pixel values (or slight variations, i.e. very few different Paeth values) might cause large overestimations of the compressed size of a compressed PNG file.
Then, we use use this Cardinality indicator in our model, with different variations (Paeth or raw image, with or without separation between RGB channels...)