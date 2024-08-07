Deep Learning - 
==============================

Restructuring your ML code for production deployment

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
        │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── model            <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reference         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── feature       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── model         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │



--------
Steps to run the program:
1. From VS Code UI, create a new Terminal by clicking ... and choose New Terminal from the menu 
2. Go to Terminal and type 'python main.py' and enter 



Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"