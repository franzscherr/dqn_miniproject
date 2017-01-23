# als-miniproject CNN
The CNN training can be done by calling python main.py

## Train CNN on MNist dataset
The training of the CNN can be done based on the MNist dataset by adjusting a single variable in the main.py script.

Therefore "trainMNist" has to be set to True. 
The script comes with already pretrained weights. If the line 46 is commented in and 45 is commented out, the algorithm is initialized with the pretrained weights. The algorithm will continue training from this step.

## Train CNN on ATARI game
To train the CNN on a ATARI game the variable "trainMNist" has to be set to false. The game to train on can be adjusted in line 31.
