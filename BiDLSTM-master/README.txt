Bi-direction LSTM README: 


Credit for the bulk of this code goes to its author - 
https://github.com/Sentimentron/Dracula

You will Need The Following Packages:
TensorFlow
Theano 
unicodecsv

Note - It is highly suggested that you use the following data science 
AWS instance to avoid having to install TensorFlow/Theano yourself:

'Deep Learning AMI with Source Code (CUDA 8, Amazon Linux)'

To train the model: 

1. Open ~/BiDLSTM/lstm.py
2. Edit lines 180-182 to refer to the files from the One Drive Directory 
   (Can be found one directory up in the overall README)
3. Edit line 240 in lstm.py to reflect the correct path to the testing 
   file from the One Drive directory
3. Run python lstm.py --words 1 (python 2.x)

To test the model (assumes you have run the training step above):

1. Run python lstm.py --words 1 --model lstm_model.npz --evaluate
