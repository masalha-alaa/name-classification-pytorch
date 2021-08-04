# Name Classification

## Introduction

In this project I attempt to predict the language of origin of given human names, all written in English characters.
The dataset is taken from [this GitHub repository](https://github.com/spro/practical-pytorch/tree/master/data/names), which tackles the same problem. Since I think there are some issues with the said repo, I decided to implement the project on my own from scratch.
Here I present a quick overview of the project. For a complete walkthrough, including the code, please head to my notebook [name_classification_rnn.ipynb ](https://github.com/masalha-alaa/name-classification-pytorch/blob/master/name_classification_rnn.ipynb).

## Dataset
The dataset consists of 16 files<sup>*</sup>, each belonging to a different langauge, and in each file there is a set of names belonging to the file's language.
This is how the data is distributed:
|label     |count|
|----------|-----|
|Arabic    |2000 |
|Chinese   |268  |
|Czech     |519  |
|Dutch     |297  |
|English   |3668 |
|French    |277  |
|German    |724  |
|Greek     |203  |
|Italian   |709  |
|Japanese  |991  |
|Korean    |94   |
|Polish    |139  |
|Portuguese|74   |
|Russian   |9408 |
|Spanish   |298  |
|Vietnamese|73   |

As it can be seen, the data is hugely imbalanced. Hence I included a code snippet to randomly select a defined number of samples per language. Also, to deal with this further, I provided the loss function with custom class weights to account for the data imbalance problem. I tried 2 versions of custom weights:
1. The inverse of the class's proportion in the data set.
2. sklearn's [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html), which estimate class weights for unbalanced datasets, using the formula:  
`number_of_samples / (number_of_classes * bincount(data))`.  
The two methods ended up with pretty much similar results.

_<sup>*</sup> originally the dataset had contained 18 files, but I deleted Scottish and Irish, since they're actually English, and we have enough English names in the dataset._

## Data split
I split the data to 80% training and 20% test sets.

## Encoding
To encode the labels from strings to numbers, I used [sklearn's LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html), which creates a 2-way mapping between labels and encodings.
To vectorize and encode the name characters, I first normalized the characters to their ascii english characters using the library [unidecode](https://pypi.org/project/Unidecode/). For example, the name "François" becomes "Francois". Although this might make it harder for the network (becaues it fades away critical hints to the language of origin), it makes it more generic and robust. After normalization, I used PyTorch's one_hot function to perform a on-hot encoding on the characters. For example, the name **"slim shady"** is converted to:

|' '|a  |b  |c  |d  |e  |f  |g  |h  |i  |j  |k  |l  |m  |n  |o  |p  |q  |r  |s  |t  |u  |v  |w  |x  |y  |z  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|1  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |


## Model
I tried 4 versions RNN:
1. Regular Vanilla RNN from PyTorch.
2. Simple custom Vanilla RNN.
3. LSTM.
4. GRU.

The loss function, learning rate, and optimizer were the same for all models:
* Loss function: [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) (Negative Log Likelihood Loss)
* Learning rate: 0.005
* Optimizer: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

## Training
Since this is a character-based RNN, the input needs to be one character at a time (hence the one hot encoding). So for each name, I fed the model with one character at a time, with the language of origin provided as the label.  
I let the networks train for 30 epochs on the training set, and after every epoch evaluate on the test set. Here is the GRU's progress log:

```
GRU(
  (rnn): GRU(59, 128, batch_first=True)
  (fc): Linear(in_features=128, out_features=16, bias=True)
  (softmax): LogSoftmax(dim=1)
)
2021-08-04 10:28:28.268167

Epoch 0: Training Loss: 1.5416, Validation Loss: 1.3454
Epoch 1: Training Loss: 1.2513, Validation Loss: 1.1672
Epoch 2: Training Loss: 1.1139, Validation Loss: 1.0701
Epoch 3: Training Loss: 1.0328, Validation Loss: 0.9980
Epoch 4: Training Loss: 0.9792, Validation Loss: 0.9472
Epoch 5: Training Loss: 0.9353, Validation Loss: 0.9270
Epoch 6: Training Loss: 0.9029, Validation Loss: 0.8900
Epoch 7: Training Loss: 0.8713, Validation Loss: 0.8717
Epoch 8: Training Loss: 0.8429, Validation Loss: 0.8477
Epoch 9: Training Loss: 0.8176, Validation Loss: 0.8317
Epoch 10: Training Loss: 0.7933, Validation Loss: 0.7901
Epoch 11: Training Loss: 0.7640, Validation Loss: 0.7762
Epoch 12: Training Loss: 0.7419, Validation Loss: 0.7632
Epoch 13: Training Loss: 0.7200, Validation Loss: 0.7503
Epoch 14: Training Loss: 0.6995, Validation Loss: 0.7188
Epoch 15: Training Loss: 0.6742, Validation Loss: 0.6969
Epoch 16: Training Loss: 0.6582, Validation Loss: 0.6841
Epoch 17: Training Loss: 0.6364, Validation Loss: 0.6775
Epoch 18: Training Loss: 0.6179, Validation Loss: 0.6661
Epoch 19: Training Loss: 0.6012, Validation Loss: 0.6687
Epoch 20: Training Loss: 0.5855, Validation Loss: 0.6546
Epoch 21: Training Loss: 0.5684, Validation Loss: 0.6547
Epoch 22: Training Loss: 0.5547, Validation Loss: 0.6417
Epoch 23: Training Loss: 0.5440, Validation Loss: 0.6151
Epoch 24: Training Loss: 0.5255, Validation Loss: 0.6114
Epoch 25: Training Loss: 0.5134, Validation Loss: 0.6224
Epoch 26: Training Loss: 0.5006, Validation Loss: 0.6087
Epoch 27: Training Loss: 0.4881, Validation Loss: 0.6194
Epoch 28: Training Loss: 0.4764, Validation Loss: 0.5973
Epoch 29: Training Loss: 0.4641, Validation Loss: 0.5920

Training Time: 0:30:09.353618
```
![gru training](https://user-images.githubusercontent.com/78589884/128221368-b753cd28-e299-44ee-869c-c0b1904a63d5.png)

As can be seen from the plot, the network keeps learning until reaching a loss of less than 0.60 and shows an intent to start converging (I didn't let it continue for lack of time and resources).

The plots for the other networks can be seen in the notebook mentioned above. GRU had the best results, however.

In addition, I printed some random predictions that the GRU network made on the validation set after finishing the training process:
|Name       |Label      |Prediction|Result    |
|-----------|-----------|----------|----------|
|Atlanov    | Russian   | Russian  | CORRECT  |
|Bekhterev  | Russian   | Russian  | CORRECT  |
|Dobroslavin| Russian   | Russian  | CORRECT  |
|Dougherty  | English   | English  | CORRECT  |
|Abano      | Spanish   | Italian  | INCORRECT|
|Jalybin    | Russian   | Russian  | CORRECT  |
|Vaca       | Czech     | Italian  | INCORRECT|
|Peter      | German    | German   | CORRECT  |
|Vingilevsky| Russian   | Russian  | CORRECT  |
|Noschese   | Italian   | Italian  | CORRECT  |
|Guirguis   | Arabic    | Arabic   | CORRECT  |
|Castro     | Portuguese| Italian  | INCORRECT|
|Her        | Russian   | Russian  | CORRECT  |
|Lejepekov  | Russian   | Russian  | CORRECT  |
|Stolarz    | Polish    | Polish   | CORRECT  |
|Tabuchi    | Japanese  | Japanese | CORRECT  |
|Gaubrich   | Russian   | German   | INCORRECT|
|Aswad      | Arabic    | Arabic   | CORRECT  |
|Gassan     | Russian   | Russian  | CORRECT  |
|Mifsud     | Arabic    | Arabic   | CORRECT  |

Finally, following is a confusion matrix of the **validation set** (opposed to the referenced GitHub repo which drew a confusion matrix of the training set, which I think is misleading).
![confusion matrix](https://user-images.githubusercontent.com/78589884/128222646-d8ee66cc-107b-4d41-8c0c-ff650486b323.png)

As can be seen in the confusion matrix, most erros occur between "close" languages, such as Chinese and Korean, or Spanish and Italian.

## Conclusion
In conclusion, I have successfully created an RNN network which classifies human names to their language of origin, with satisfying low loss and high accuracy.
