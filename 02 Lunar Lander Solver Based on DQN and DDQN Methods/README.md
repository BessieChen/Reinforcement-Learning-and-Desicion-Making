## This is the instruction of how to use the .py file and .txt file in Project_02 folder:

### 1.I wrote .py file using python version 2.7

### 2.The packages I used are: 
gym, random, tempfile, numpy, collections, keras, os, datetime, matplotlib

Just common packages that are frequently used in ML assignments.

### 3. There are 8 mini folders in Project_02 folder, they are the raw inputs of my .py file. 

You should download all .txt file in these 8 folders if you want to know how my .py file works. 

(Note that the file path can be modified in .py file if necessary!)

Especially when you want to:

A. Test the model performance through the model weights which have saved in .h5 files.

B. Plot the reward and average reward of training experiments with different hyper-parameters

C. Plot the reward and average reward after testing on specific models.

### 4. There are only two python file in Project_02 folder:

project02_main.py, in which you can:

A. run my default DQN model since all hyper-parameters are assigned with default value.

B. run the DDQN model by setting the ddqn = True

C. test specific model by setting test = True and giving .h5 file which saved the model weights

D. playing with the hyper-parameters

project02_plot.py, in which you can:

A. Plot the reward and average reward of model with different hyper-parameters

B. Plot the reward and average reward after testing on specific models

Don't forget to download those 8 mini folders before you run project02_plot.py! Thanks!!
