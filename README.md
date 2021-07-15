# ImbalancedRegressioninDataStreams

# Relevance based Pre-processing Strategies for prediction of rare events in an imbalanced regression setting.

The Static and Dynamic Relevance module in Python is used to predict rare events in a target continous variables when the time component is associated. Basically your data needs to be time-series or have dates associated with it. The various pre-processing strategies helps to mitigate the problem of Imbalanced data in a regression setting. 

**Authors:**

Prof Dr. Paula Branco, pbranco@uottawa.ca

Mayukh Bhattacharjee, mayukhofficial12@gmail.com

Durga Prasad Rangavajjala, drang041@uottawa.ca

# Setting up the necessary things to run the code:

For this project we have used Fortran code (this is just for the calculation of the Relevance values also called Phi). You need not do anything with that module, and the relevance will be calculated automatically. 
For the successful execution of the code, you need to follow the steps.

**->** Install the MINGW 64/32 (depending on your PC specification). You can also download the file that is in this repo but that is for 64 bit version.

**->** After the installation process, Set the environment variables- Path: C:\MinGW\mingw64\bin

Congrats! you have completed the basic steps.

To run the code on your datasets, you need an IDE for this particular project to parse and run the .dll files contained in the PhiRelevance module, for example (PyCharm, MS Visual Studio Code). This won't run on Anaconda Navigator or Google Colaboratory. 




