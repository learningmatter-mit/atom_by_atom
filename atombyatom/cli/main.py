# cli/main.py
import argparse
import subprocess
import os
import gdown
from atombyatom.data import DEFAULT_DATA_PATH, DEFAULT_MODELS_PATH, DEFAULT_RESULTS_PATH, DEFAULT_ANALYSIS_PATH

url_dict = {'bulk_dos': 'https://drive.google.com/uc?id=1mLQYkamCFf-68FE_crt476mShs-zOue1'}

class CLICommand:

    def download(self, dataset):

        '''
        Download dataset from Google Drive
        '''
        
        url = url_dict[dataset]
        output_file = DEFAULT_DATA_PATH + dataset + '.json'

        # download file from url to output
        gdown.download(url, output_file, quiet=False)

    def run(self, model, dataset):

        '''
        Run the model on the dataset
        '''

        # results directory
        results_dir = DEFAULT_RESULTS_PATH + model + '/' + dataset

        # if results directory does not exist, create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # get run.py from models folder
        run_file = DEFAULT_MODELS_PATH + model + '/run.py'
        dataset_file = DEFAULT_DATA_PATH + dataset + '.json'
        data_cache = results_dir + '/' + dataset + '.cache'

        # call run.py with arguments
        subprocess.call(['python', run_file, '--data', dataset_file, '--data_cache', 
                         data_cache, '--results_dir', results_dir])
        
    def analyze(self, model, dataset, site_prop):

        '''
        Analyze the results of the model on the dataset
        '''

        # results directory
        results_dir = DEFAULT_RESULTS_PATH + model + '/' + dataset
        analysis_file = DEFAULT_ANALYSIS_PATH + 'plot_parity_bulk.py'

        # call analysis script with arguments
        subprocess.call(['python', analysis_file, '--model', model, '--dataset', dataset, '--site_prop', site_prop, '--results_dir', results_dir])
        



def main():
    # Create a top-level parser
    parser = argparse.ArgumentParser(prog='atombyatom')

    subparsers = parser.add_subparsers(dest='command')

    # Create a parser for the 'download' command
    download_parser = subparsers.add_parser('download', help='download data')
    download_parser.add_argument('--dataset', default='bulk_dos', help='Dataset to download')

    # Create a parser for the 'run' command
    run_parser = subparsers.add_parser('run', help='run model')
    run_parser.add_argument('--model', default='per-site_cgcnn', help='Model to run')
    run_parser.add_argument('--dataset', type=str, default='bulk_dos', help='Dataset to run on')

    # Create a parser for the 'analyze' command
    analyze_parser = subparsers.add_parser('analyze', help='analyze results')
    analyze_parser.add_argument('--model', default='per-site_cgcnn', help='Model to analyze')
    analyze_parser.add_argument('--dataset', type=str, default='bulk_dos', help='Dataset to analyze')
    analyze_parser.add_argument('--site_prop', type=str, default='bandcenter', help='Site property to analyze')

    args = parser.parse_args()

    # Initialize the CLICommand class
    command = CLICommand()

    # Based on the command name, call the appropriate function
    if args.command == 'download':
        command.download(dataset=args.dataset)

    elif args.command == 'run':
        command.run(model=args.model, dataset=args.dataset, site_prop=args.site_prop)

    elif args.command == 'analyze':
        command.analyze(model=args.model, dataset=args.dataset, site_prop=args.site_prop)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
