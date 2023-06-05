import gdown
import argparse
from atombyatom import __root_dir__
from atombyatom.data import DEFAULT_DATA_PATH


url_dict = {'bulk_dos': 'https://drive.google.com/uc?id=1mLQYkamCFf-68FE_crt476mShs-zOue1'}

def download_data(dataset):

    url = url_dict[dataset]
    output_file = DEFAULT_DATA_PATH + dataset + '.json'

    # download file from url to output
    gdown.download(url, output_file, quiet=False)

