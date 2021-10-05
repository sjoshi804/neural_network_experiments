from datetime import datetime 
import os 
from pprint import pformat

class BaseExperiment():
    def __init__(self, config):
        self.config = config
    
    def results(self):
        # Create directory for results
        current_time = str(datetime.now())
        os.mkdir(current_time)
        # Write config to file
        with open("config.txt", "w") as config_file:
            config_file.write(pformat(self.config))
        # Write results
        self.write_results(current_time)