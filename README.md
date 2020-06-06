## Convolutional Neural Network for Surgical mask detection

This project was actually a competition on kaggle between the students in my year.

### There were 2 phases:

1) We develop our models, train them on 8000 training examples, test them on 1000 validation examples and then use them to predict on 3000 test examples. We are ranked by our accuracy on a subset of 33% of those test examples. At the end of this phase I was ranked 1st in the competition with an accuracy of 73.22%.
2) We are ranked by our accuracy on the whole test set (3000) examples. After this last phase I was still ranked 1st in the competition with an accuracy of 70.76%.


### Description:

In the surgical mask detection task, participants have to discriminate between utterances with and without surgical masks. The system could be useful in the context of the COVID-19 pandemic, by allowing the automatic verification of surgical mask wearing from speech.


### Data processing:

We were given audio samples, 1 second each: 8000 samples for training, 1000 samples for validation and 3000 samples for testing. The training and validation samples were labelled.
I used the python package ‘scipy.io’ for reading the data from the audio files. Each 1 second audio file was converted to a 16.000 features vector, representing the sound amplitude measured 16.000 times over 1 second. One audio file converted with the noted approach looks like this:

![audio-amplitude](https://user-images.githubusercontent.com/48453930/83947613-cf62cd00-a820-11ea-872e-bddf1116d267.png)

After converting each audio file to a feature vector, my sets looked like this:

Training-set: 8000x16000

Validation-set: 1000x16000

Test-set: 3000x16000


The next step was applying data augmentation to the training set. I have used 3 different approaches, and after extensive testing, using only the first 2 gave the best results:
1) Add random noise to an audio: For each training example, I have created a new artificial example with the following approach: for each feature, generate a random number with normal distribution of mean 0 and standard deviation 0.2 (this number may not be the perfect choice, but while testing I concluded that the noise added is not too much and not too small) and set the corresponding feature of the new example as: original_feature_value * (1 + generated_number).

2) Shift sound of audio: For each training example, I have created a new artificial example with the following approach: generate a random number with a uniform distribution between 1600 and 4000. The feature vector for the new example is the feature vector of the original example shifted X positions to the right, where X is the generated number. (the right side of the feature vector gets moved to the beginning of the vector, it is not lost).

3) Add random noise AND shift audio: both of the previous methods combined.

I have used the methods multiple times for increasing the training set size, 90.000 examples being the biggest training set I could create before my session crashed on google colab because of running out of RAM. Training sets of 24.000, 56.000 and 90.000 gave good results that I took into consideration for my final submission.


### Model used:

The model I used was the convolutional neural network, built in the Keras programming framework. I was already familiar with both convolutional neural networks and Keras from the deep learning course I have completed recently and from another project I’ve been working on in the last weeks.

The layers I used are as follows: 1D Convolution, 1D Maxpooling, Flatten, Dense. I have also used batch normalization to keep the data normalized and improve the speed and performance of the model, l2 regularization and dropout to avoid overfitting.

During the competition I have trained a lot of different models, with different hyperparameter choices, and I will present only one of them. For the final submission I have combined the predictions of multiple models (6) to make sure my predictions are consistent. The model I will present next is one of those 6 models.


### Architecture:

1) Conv1D with 16 filters of size 15, stride 3, padding ‘same’, l2 regularization with lambda=0.0045, batch normalization and ReLU activation.
2) Maxpooling1D with a pool size of 2 and dropout with rate of 0.1.
3) Conv1D with 16 filters of size 15, stride 3, padding ‘same’, l2 regularization with lambda=0.0045, batch normalization and ReLU activation.
4) Maxpooling1D with a pool size of 2 and dropout with rate of 0.1.
5) Conv1D with 32 filters of size 15, stride 3, padding ‘same’, l2 regularization with lambda=0.0045, batch normalization and ReLU activation.
6) Maxpooling1D with a pool size of 2 and dropout with rate of 0.1.
7) Conv1D with 32 filters of size 15, stride 3, padding ‘same’, l2 regularization with lambda=0.0045, batch normalization and ReLU activation.
8) Maxpooling1D with a pool size of 2 and dropout with rate of 0.1.
9) Flatten
10) Dense with 64 neurons and l2 regularization with lambda=0.01, ReLU activation and dropout with rate of 0.2
11) Dense with 32 neurons and l2 regularization with lambda=0.01, ReLU activation and dropout with rate of 0.2
12) Dense with a single output neuron, sigmoid activation.

Using 4 convolutional layers, each followed by a maxpooling layer gave me much better results than any other architecture I tried, including 2 of each, or 4 convolutions and a maxpooling after every 2 convolutions. 

The filter size of 15 may seem a bit too large, but I have tried with 3, 5, 10 or even larger values like 20, 25 and they all gave much worse results. I have even varied the filter sizes from one layer to another, like using 15 for the first 2 convolutions and 10 or 5 for the last 2, but the results were worse. 

The ReLU activation function for all but the last layer was an easy choice, as it is the most popular in the deep learning community. Used sigmoid activation for the output layer as this is a binary classification problem.

The number of filters in each convolutional layer and each dense layer was a matter of experimentation. I have tried a lot of different values, smaller and bigger (even much bigger, like 512 neurons in the dense layers), but smaller values seem to work better, perhaps because our 16.000 features get converted to a much smaller number really fast, as every convolutional layer divides their number by 3 (stride is 3) and every maxpooling layer divides their number by 2. Also the amount of data we have (without augmentation) is pretty small I’d say, and I’m not sure my methods of data augmentation are very good, they might help only by a small margin. The Flatten layer has only 384 neurons.

The regularization and dropout came naturally as my previous models had too much variance, denoted by the big difference in accuracy between training and validation set (like 20-30% difference, 90% vs 65% accuracy maybe). I have tried a lot of combinations with those parameters and the model was very sensitive even to slight increases, that’s why the number 0.0045 may seem a little odd for the l2 regularization, or the 0.1 dropout rate may not seem to be a big deal, but it certainly makes the difference.


### Training:

The optimizer I used is Adam, as I was more familiar with it and I know it is a good choice most of the time and the default parameters (from the paper) are a good choice and in most problems require no tuning. It uses learning rate decay to decrease the learning rate after training for more epochs, to get as close as possible to the cost function’s local minimum.

The cost function is binary cross entropy and the evaluation metric accuracy.

I have trained the model on the training data (augmented) for like 100 epochs and evaluated on the validation data. The mini batch size I used is 32, and I have shuffled the training set before each epoch.



### Statistics on validation data:

![confusion-matrix](https://user-images.githubusercontent.com/48453930/83947785-ddfdb400-a821-11ea-84af-7d277bcc5df4.png)



### Final submission:

I have taken into consideration the predictions of 6 of my trained models. Their accuracies on kaggle (on the test set) ranged from 69% to 71.6%. My prediction for each example in the test set was an average of the predictions of those 6 models (before rounding to 0/1 values). The combined accuracy, on kaggle, is 73.22% and the final predictions should be more consistent than say if I used only the model with 71.66% accuracy. 

I am int the **1st** place in the competition on the public leaderboard. When the final standings are revealed, we will be ranked by our model’s accuracy on the whole test set (3000 examples) instead of only a subset (1000 examples).

![kaggle_before](https://user-images.githubusercontent.com/48453930/83947930-cd017280-a822-11ea-9850-1d42bcb06974.png)


### After the final standings for the competition have been revealed:

I am still ranked **1st** in the competition with an accuracy of 70.76%!

![kaggle_after](https://user-images.githubusercontent.com/48453930/83947968-06d27900-a823-11ea-9143-458110773c32.png)