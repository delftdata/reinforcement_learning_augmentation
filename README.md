# Reinforcement_learning_augmentation
Modified code and experiments from the "Feature augmentation with reinforcement learning" paper

The repository is structured as follows:
- `classification-based` contains an implementation of the algorithms that is applicable to datasets with a classification task;
- `regression-based` contains an implementation of the algorithms that is applicable to datasets with a regression task;
- `results` contains the results of the findings of the two approaches in `csv` formats and also a Jupyter notebook that visualizes the results;

Both `classification-based` and `regression-based` folders have the same structure:
- `main.py` - main file that can be used to start both the feature selection approaches. Note that this file also contains code to fetch data from other repositories if it is not locally available;
- `Agent_MAB.py` - contains the actual implementation of the multi-armed-bandit feature selection algorithm;
- `Agent_RL.py` - contains the actual implementation of the reinforcement-learning feature selection algorithm;
- `download_data.py` - used to fetch data from Google's BigQuery. Note that an API key is required to run the code in that file. Therefore, you should first acquire an API key to fetch information from Google's cloud service before you can download data from there;
- `Environment_MAB.py` - location to set algorithm-dependent environment such as hyperparameters and datasets to use;
- `Environment_EL.py` - location to set algorithm-dependent environment such as hyperparameters and datasets to use;
- `MAB_main.py` -  main file to start the multi-armed-bandit-based algorithm. Note that this is the main place, where you can set algorithm hyperparameters and also put paths to datasets to be used for augmentation.
- `RL_main.py` - main file to start the reinforcement-learning-based algorithm. Note that this is the main place, where you can set algorithm hyperparameters and also put paths to datasets to be used for augmentation.
- `requirements.txt` - all dependencies required to successfully run the code; 