You need to install two packages. Please use these commands:
sudo env "PATH=$PATH" python3 -m pip install nltk
sudo env "PATH=$PATH" python3 -m pip install tqdm


This folder contains the code for the Decomposition Attention Model.

To train the model run:
python3 pair_classifier_train.py

To obtain the predictions run:
python3 pair_classifier_infer.py

To make any changes to the hyperparameters or to vary the number of layers in the model , please make changes to pair_classifier_model.py.
Eg : To vary the number of hidden units in the first feed forward layer from 300 to any value (Eg: 200), change the number of units in line 19 and 22.
F1a = tf.layers.dense(inputs=self.a, units=200, activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=None, name='F1')