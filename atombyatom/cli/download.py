import gdown
from atombyatom.utils.download import download_data

class CLICommand:

    '''
    Download data from Google Drive
    '''

    def run(self, args):
        download_data(args)

 

