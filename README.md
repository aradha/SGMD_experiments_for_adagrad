Experiments for the paper Linear Convergence of Generalized Mirror Descent with Time-Dependent Mirrors.

Outline of files:

1. main.py - runs full batch Adagrad experiments for the noiseless linear regression setting. 

2. stochastic_main.py - runs stochastic Adagrad experiments for the noiseless linear regression setting. 

3. neural_network_files - contains code to run full batch Adagrad experiments for over-parameterized neural network in the noisy regression setting.  Just use main.py to run the code and use the options in options_parser to add in details for number of examples and number of dimensions for the regression setting.  The model can be changed in neural_model.py and the training method can be changed in trainer.py.   

Dependencies:

1. numpy
2. matplotlib
3. pytorch 0.4.1 (cuda enabled)
