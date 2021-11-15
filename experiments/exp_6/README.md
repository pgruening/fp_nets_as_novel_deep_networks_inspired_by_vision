# Experiment 6: Are hyper-selective neurons more robust to noise?

Use already trained models. For Cifar, create a test dataset that uses jpeg compression. Are the results the same? How is the test error evolving for different compression rates?

## Abstract

Are hyper-selective structures in a CNN more robust to noise?

## Introduction

If our FP-nets are more robust to noise, it confirms some of the findings in [Olshausen et al.](https://jov.arvojournals.org/article.aspx?articleid=2772000), and add another advantage to using FP-nets.

## Experiment

We use the Cifar-10 test set. For each image, we create n other jpeg compressed version, i.e., noisy images. A robust model should, regardless of noise, consistently predict the exact and correct class for an image.
We look at the standard test set score, how does it change with these additional data.
Furthermore, we count how often a model 'changes its mind' when an image is altered by the compression.

## Hypothesis

We assume that the test error is increased for all models—however, less for FP-nets. Additionally, FP-nets change the class of one image less often than standard models.

## Name at least four possible bugs. Did you check them before running?

* The wrong image normalization is used
*We use the same transformations as for the test-set in ds_cifar.*

* The jpeg compression alters the images in a strange way
*The data were checked with Tensorboard.*

* OpenCV reading/writing RGB/BGR issues
*We don't really need to use OpenCV writing and reading operations. The jpeg compression does not seem to alter the ordering of the channel. Any IO is done with NumPy.*

## Hyperparameters

We chose ten different quality values for each image: 100, 90, 80, ..., 20, 10. 

## Name at least two tests/outcomes that indicate something is wrong. Did any of this happen?

* The model's initial error values are different than the ones in the logs

*The error plots were compared; they look identical.*

* There is no prediction change with different compression rates.

*Depending on the compression ratio, the error can increase drastically.*

## Results

Here are some examples of the input images. The number on the top of each image indicates the quality of the image (100 - compression ratio). The top left image is the original image.

<img src="images/examples/00.png">
<img src="images/examples/01.png">
<img src="images/examples/03.png">
<img src="images/examples/05.png">

Given a quality below 50%, many images are hard to classify –even for a human. Although, vehicles, ships, and planes may be easier to spot than animals. 

We compute the test error (in percent) for all images with a specific quality. We compare the initial error (left, similar to Experiment 0) to the error when all images are jpg compressed with 90% quality (right). The filled areas denote the minimum and maximum values out of 5 models trained with different seeds.

<p float="left">
    <img src="images/results/er_0.png" style="width:45%">
    <img src="images/results/er_1.png" style="width:45%">
</p>

The x-axis shows the number of blocks used in the network. The mean error for the largest FP-net (num blocks=9) increases from 5.8 to 10.2. The increase is even higher for the PyramidNet (6.3 vs. 11.2). Remarkably, the ResNet's error increases slower (6.7 vs. 10.7). The error further increases with decreasing image quality.

<p float="left">
    <img src="images/results/er_2.png" style="width:45%">
    <img src="images/results/er_3.png" style="width:45%">
</p>

With a quality of 40% - 50%, the ResNet's performance is on par with the FP-net, in some configurations even better.

<p float="left">
    <img src="images/results/er_4.png" style="width:45%">
    <img src="images/results/er_5.png" style="width:45%">
</p>

The mean error of each model is given in this table. 'er_i' denotes that the min test error on a dataset with (100 - i*10)% quality was computed (0->100%, 1->90%, ...). 


| model_type     |   Error Q: 100 |   Error Q: 90 |   Error Q: 80 |   Error Q: 70 |   Error Q: 60 |   Error Q: 50 |   Error Q: 40 |   Error Q: 30 |   Error Q: 20 |   Error Q: 10 |
|:---------------|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| CifarJOVFPNet  |           5.85 |         10.12 |         14.23 |         18.02 |         21.12 |         23.82 |         26.89 |         32.38 |         40.83 |         57.93 |
| CifarPyrResNet |           6.23 |         10.97 |         15.29 |         19.09 |         22.64 |         25.5  |         29.24 |         34.52 |         43.22 |         59.62 |
| CifarResNet    |           6.81 |         10.79 |         14.83 |         18.16 |         21.56 |         24.2  |         27.73 |         32.27 |         40.94 |         57.84 |

Of course, looking at this metric, the FP-net may have a clear advantage since it performs better on the initial test set. Thus, we evaluate another metric: the number of changed classes. Given the predictions of the original test-set *P_0* and the predictions on a noisy test-set *P_i*, we count how often the network changed the prediction. We report this number in percent; Cifar-10's test set contains 10000 images, if, for a specific network, in *P_i* 100 predictions differ from *P_0*, the network "changed its mind" on 1% of the data.

Here are plots comparing the percentages of changed predictions given the model type and the number of blocks used in the model. The quality is from left to right, top to bottom 90%, 80%, 70%, and 60%.

<p float="left">
    <img src="images/results/nc_1.png" style="width:45%">
    <img src="images/results/nc_2.png" style="width:45%">
</p>
Larger models tended to be more robust, but the data show that, even with 90% quality, each network changed 8% to 10% of the prediction classes. This value increased to 12%-15% for 80% quality. For higher rates, the FP-net was more robust than the two baseline models. However, with decreasing image quality, the ResNet yielded comparable results.

<p float="left">
    <img src="images/results/nc_3.png" style="width:45%">
    <img src="images/results/nc_4.png" style="width:45%">
</p>

The next table shows the mean percentage of prediction changes for all models with 9 blocks:

| model_type     |   NCC Q: 90 |   NCC Q: 80 |   NCC Q: 70 |   NCC Q: 60 |   NCC Q: 50 |   NCC Q: 40 |   NCC Q: 30 |   NCC Q: 20 |   NCC Q: 10 |
|:---------------|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| CifarJOVFPNet  |        7.69 |       12.48 |       16.7  |       19.87 |       22.7  |       25.89 |        31.4 |       40.06 |       57.77 |
| CifarResNet    |        8.18 |       12.76 |       16.73 |       20.06 |       22.89 |       26.61 |        31.3 |       40.2  |       57.47 |
| CifarPyrResNet |        8.74 |       13.5  |       17.59 |       21.26 |       24.29 |       28.19 |        33.6 |       42.56 |       59.33 |



## Discussion

From those results, we can derive a few points:

1. Even small changes in the images (original vs. 90% quality) can lead to about 10% of predictions being changed by the CNNs, increasing the test error up to 4 points (from 5.8 to 10.1 for the FP-net). 

2. The FP-net is more robust to more minor changes in quality  80%-90%; for lower values, the FP-net and the ResNet are comparable.

3. The pyramid-net is far more susceptible to noise. 
