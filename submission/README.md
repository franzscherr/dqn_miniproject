# Here is our implementation of DQN

## Test
Run python test.py

This will load trained parameters for the value advantage splitted Q-network and proceed to play indefinitely long pong until you SIGINT it.

## Train
Run python main.py [log_and_model_and_config_dir]

This will trigger the Q learning algorithm to start. If the directory is given it will continue to train from the data found. If not given it will create this directory self and constantly save the model and log.
Careful: logfile can grow big

## CNN
In the subfolder CNN you will find all scripts and files necessary to evaluate our CNN. A sub readme is also included.

## makeplot.py
Run this with the argument as the directory to create the plot from. Will not work for the included trained directory since we stripped the quite big log file.

## main_pacman.py
Same structure as main.py but would learn to play pacman. But this is not proved as it requires a big amount of computational power.
