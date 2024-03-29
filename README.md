This is a Carnatic Music Raga classification project.
The img_result contains fft of 10 recordings each in begada and varali.

# Issues
1. Cluster the images using t-SNE and observe the utility of this endeavor 
2. Perform k-means clustering using the fft feature directly and report the result
after 100 iterations, averagd over 10 times.


## Install Dependencies

## How to run

## Datasets
CompMusic varnams dataset with 28 recordings (mp3 format)

- Every music file is split into the below specified duration and a spectrogram for every clip is produced by `librosa.display.specshow()`.

- A CNN takes the spectrogram as input and is used to produce logits as output. These logits are accumulated or summed up for all the short clips of a single music file. Finally `softmax()` is applied to this accrued logits values. This is the output of forward propagation of the neural network.

- An alternative is to artificially augment the dataset and perform trainingon the short clips. Downside: A different starategy will have to be adopted during inference, validation and testing.

- `categorical_cross_entropy()` loss function is used as the loss function. Adam is used as the optimiser.

## Hyper-parameters 
- Architecture of CNN: 
[Architecture of the network](/results/arch.png)


- Duration of every clip:
- Overlap between segments used in `scipy.signal.stft`
- Learning Rate: 0.01
- Number of Epochs during training: 5 per training sample
- Optimiser: Gradient Descent Optimiser

## Apprehensions
- The idea of accruing the answer by looking at short music clips is not compatible with the existence of `gamakas` or inter-frequency swaying of the voice. Moreover, the short clips might not be representative of the raga. Another possible alternative is to use an LSTM instead of taking the overall statistic by repeatedly applying the CNN.

## Future Work
1. Non-linear optimal representations of every raga needs to be visualised and verified with our intuition of closeness or allied ragas.
2. Explore the information rates of different ragas and investigate it's relationship with the concept of `rakti ragas`. 
Ref: https://advances.sciencemag.org/content/5/9/eaaw2594

