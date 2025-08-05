# GaussianNoiseU8-testing
This project contains a number of utilities to test different implementation candidates that aim to add gaussian noise to an image with the ``uint8`` data type. To reproduce results, one must have a ``Python 3.13`` environment and must install the necessary libraries:

```
pip3 install -r requirements.txt
```

## Implementations tested
The transform essentially creates a tensor with the same dimension as the input (following a normal distribution of mean 0 and standard deviation 1), then multiplies said tensor by a positive scalar (sigma), adds an additional scalar (the mean), and finally adds that result element-by-element to the input image. The original implementation expects the input to be of the data type ``float``, with a maximum possible value of 1 and a minimumm possible value of 0.

To make this transform work on ``uint8`` images, we need to transform said floating-point, ``[0, 1]`` range noise to the integer, ``[0, 255]``-ranged format. We've tested two ways to do this:
1. The first method transforms the input to the ``float`` data type, adds the noise and transforms the output back into ``uint8``.
    * Please note that we *don't* necessarily have to move the input image's representation range from ``[0, 255]`` to ``[0, 1]`` and back, as those conversions are performed per-pixel and can be slow on big images. Instead, as the noise is going to be multiplied by the ``sigma`` parameter anyways, we can use the ``sigma * 255`` as a coefficient instead (and then add ``mean * 255``), thus saving two floating-point array-wide operations.  
2. The second method involves less floating-point operations, opting to convert both the noise and the input image to an intermediate data type (``int16``) to then perform the addition and finally transform the result back into ``uint8``.
    * Again, we multiply the noise by ``sigma * 255`` and add ``mean * 255`` before transforming the output to ``int16``.
    * Using a signed, bigger integer dtype is essential here, as we need to both cover the legitimate ``[-255, 255]`` range that the noise tensor would theoretically generate, and also have a margin to be able to clamp pixels that might lie outside said range. ``in16`` offers a range of ``[-32_768, 32_767]``, which is more than enough for our use case.
    * We have also tested some other configurations for this setup, like converting the noise to ``int16`` before adding the mean (thus performing addition of ints instead of floats), rounding the result (to have more accurate results at a performance cost) and performing implicit conversions of data types instead of an explicit ones.

## Benchmarking
Benchmarks are performed with the ``torch.utils.benchmark`` utility, ensuring high-precision, low-variance results. All relevant configurations are explained the ``benchmark.py`` script, just run the following for help:

```
python3 benchmark.py -h
```

A handy test suite can also be ran directly on the ``run.sh`` shell script.

All results get logged on ``yaml`` files and are sorted from fastest to slowest on average. Their filenames each indicate the respective image size used for testing (in side-length units, so the square root of the total number of pixels). Also, the results of some benchmarks made on personal computers (both on CPU and GPU!) are already available on the repo for convenient analysis.

## Output validation
The script ``validation.py`` analyses per-pixel differences between each GaussianNoise implementation given a set seed and deterministic algorythm usage. When you run said script, three main visualizations get done:
1. Firstly, the difference between means and standard deviations of different outputs get displayed on screen.
2. Secondly, three difference matrices get plotted, showing the average pixel-by-pixel differences between implementation outputs on the ``[0, 255]`` representation scale. Differences should be very small except for the 'float and back ``float16``' implementation, as it has utilizes a different RNG algorythm compared to the settings that utilize the ``float32`` data type.
3. Finally, a sanity-check gets performed. In it, a test image (``output_validation/noise_input.png``) gets passed to both the 'float and back' GaussianNoise transform implementation and the 'intermediate int' one, under the exact same generation seed, and the results get outputted to ``output_validation/noise_output_float.png`` and ``output_validation/noise_output_int.png``,  respectively. Results should be indistinguishable by the naked human eye.
