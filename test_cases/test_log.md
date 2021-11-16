# Test Log JOV

## Presented results

* Cifar-10 classification
* Imagenet classification: table & image
* Iso-contour plot
* Entropy
* Degree of end-stopping
* angle distribution
* gradient descent example
* adversarial examples -> image and tables\
* jpeg compression -> image and tables
* square input image

## Cifar 10 classification

The images were looked at using Tensorboard. The plot generation was read.

### Is the model plot correct?
The first Conv-Layer lacked a ReLU. From original paper, it is not entirely clear if this activation was used. However, we found it in Ydelbaye's Repo. The error was corrected in the text.


### Is the FP-block image consistent with the actual code?
Yes. This was checked. The tensorboard representation checks out, as well as the code.

### Are our models consitent with the cited repos and our previous results?
All Conv blocks used in the code were tested against the code from the original repository and the old JOV repo. The ouput is identical.
The results are comparable to the ones from the Ydelbayev repo, we are slightly better for N=3 and N=5.


### Are the numbers correctly displayed in the plots?
The number and the plot curves are consistent. The code of the plot function was reviewed.

### Are the numbers reproducible when using the models
During the jpeg and advex tests, we've obtained identical results for Q=100 and eps=0, respectively.

## ImageNet
The saved images look okay. The dataloader appears to be fine.

### Are the results reproducible?
Yes.

### What are the differences between the two used datasets?
We've found the right test time augmentation so that the numbers for each network are reproducible with both datasets.

## Modeldatabase: Entropy, Degree of end-stopping

### Are the models loaded properly?

The code has been updated to only allow strict matches. Furthermore, when loading the imagenet resnet printing some of the weights always gave the same results.

### Do the low entropy feature maps look plausible?

The statistical entropy values appear to be consistent with the entropy values over the feature maps.

### Is the entropy correctly computed?

The zero model correctly outputs zero for all acativation and entropy values.

### Are the depth values and positions right for each block?

The numbers of blocks and filters were tested and look right. Furthermore, the values that are expected to be nan are nan.

### Do the degree of end-stopping images look plausible?

Yes. High Ratios do indee show activations where the corners are very emphasized.

### Are the computations claimed in the appendix consistent with the code?

Yes the text was read carefully and compared to the code.

### Is the squared input image consistent with the appendix. For both Cifar-10 and ImageNet?

The input image was plotted and looks plausible for both Cifar-10 and ImageNet.

## Adversarial Examples
The eval code was read again.
### Is the order of FGSN and normalization correct?
An assertion statement makes sure that the input image is in [0, 1]. The Cifar dataloader has a do-not-normalize flag. The BaseModel for all Cifar models has an attribute 'pre_transform' which is used for normalization before the actual forward pass. All tested models are derived from this class. Furthermore, the original accuracies reproduce the classification results and the number of changes are zero for eps = 0.

### Do the adversarial examples look plausible?
The plotted examples that are also present in the paper are plotted directly from the evaluation.

### Are the tables computed correctly?

The code was read again. The tables are consistent with the plots.


## Jpeg compression

### Are there possibly errors due to incorrect normalization?
All images are pre-computed via an OpenCV algorithm. The algorithm is applied directly to the uint8 images and generates a set of uint8 images.

### Are the tables computed correctly?

The code was read again. The tables are consistent with the plots.



## CIGA example

### Are the formulas consistent with the text?

The code was evaluated, compared to the text and documented.


