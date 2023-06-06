import subprocess
from atombyatom.data import DEFAULT_DATA_PATH, DEFAULT_MODELS_PATH, DEFAULT_RESULTS_PATH

class CLICommand:

    '''
    Download data from Google Drive
    '''

    def run(self, model, dataset):


        # get run.py from models folder
        run_file = DEFAULT_MODELS_PATH + model + '/run.py'
        dataset_file = DEFAULT_DATA_PATH + dataset + '.json'
        data_cache = DEFAULT_DATA_PATH + dataset + '.cache'
        results_dir = DEFAULT_RESULTS_PATH + model + '/' + dataset

        # call run.py with arguments
        subprocess.call(['python', run_file, '--data', dataset_file, '--data_cache', 
                         data_cache, '--results_dir', results_dir])