## Why PNG?

We attempt to estimate the size of a PNG file, using its raw pixel data. Starting from pixel data, PNGs are first filtered (Paeth filtering) to obtain a gradient-like transformation. Then, pixel data is compressed using the lossless _Deflate_ algorithm (LZ77 class, used in zlib). For the moment, no work seems to exist to estimate size of compressed data using such algorithms.