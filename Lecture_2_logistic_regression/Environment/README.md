# Environments

This folder contains the environment that we will be working on 12/8/2017. 
Since most of you have Anaconda4.7 installed this should be pretty straightforward. 


## Installing the environment using the .yml file

Steps:

* Open a new console, cmd, git bash, etc

* Create a copy of the repository (git clone NAME_REPO)

* Enter the repository
* Go to the folder Environment

* Now run in your console:
* conda env create -f environment.yml
 
* The previous command might take some time ...
* Once it has finished, run in your console:
* source activate dl
* dl is the name of the environment

* To leave the environment use:

* source deactivate dl



# Installing the environment using the requirements.txt file 
* Open a new console, cmd, git bash, etc

* Create a copy of the repository (git clone NAME_REPO)

* Enter the repository
* Create a new environment, run in your console:
* conda create --name NAME_ENVIRONMENT python=3
* Then run:
* source activate NAME_ENVIRONMENT
* Go to the folder Environment

* Now type in your console:
* pip install -r requirements.txt

# How to make sure everything is ok?
* Run the script test.py
* If the message "You are ready for saturday's Dojo" appears, then you will be fine.