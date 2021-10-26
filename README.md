# ImageInterpretationGroup3


There are two files: network_final.py and classifier_final.py

The training and test data set must be placed in the same folder as the two python scripts. 

Run first network_final.py
 - on line 16 you can chose which dataset should be used (training or test) (run the network for both datasets separatly)
 - on line 18 the name of the output folder has to be set, this folder has to be created by hand, create seperate folders for the training and the testing data
 - on line 43 and 45 it is possible to set the numbers of patches in the grid 
 - the scirpts writes .npy vectors to the output folder

After running the script network_final.py it is possible to run classifier_final.py
- classifier_final uses the output from network_final.py
- check on line 55 and 57 that there is the same grid size as used in the network_final.py script
- on line 44 and 46 you can set the path to the training and test data ouptut from the script network_final.py
