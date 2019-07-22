# RedditFlairPredictor
A Web Application to predict Reddit Flairs

## Deployed and live at

[Heroku](http://flairpredictor.herokuapp.com/) - Reddit Flair Predictor

### Directory Structure

This is a ***Flask*** Web Application set-up for hosting on *Heroku* servers.

1. [App.py] - This is the main app outlet application
2. [requirements.txt] - Contains Dependencies
3. [Jupyter Notebooks] - Folder which contains all the scripts
4. [Helper.py] - This is the application which is called to predict flair.
5. [Procfile] - Needed to setup Heroku.
6. [Templates] - Contains all static pages.
7. [Runtime] - To point Heroku with the required python version
8. [nltk.txt] - Used to download Nltk resources
9. [Trained Data] - Contains trained data and models

### Project Execution

  1. Open the `Terminal`.
  2. Download the Repo.
  3. Ensure that `Python3` and `pip` is installed on the system.
  4. Create a `virtualenv` by executing the following command: `virtualenv -p python3 env`.
  5. Activate the `env` virtual environment by executing the follwing command: `source env/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Enter `python` shell and `import nltk`. Execute `nltk.download()` and exit the shell.
  8. Add a .env and add required environment variables.
  9. Now, execute the following command: `python app.py` and it will point to the `localhost` with the port.
  10. Hit the `IP Address` on a web browser and use the application.

### Process

Went through various documentation and refrence links to understand the complete process. 

1. Extracted the data
2. Cleaned and processed the data.
3. Selected the best model using scaled pipe.
4. Created models and chose the best model with highest accuracy

### Accuracy

0.57821721



