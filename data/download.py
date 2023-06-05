import gdown
import argparse

# take in input file name using argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='input file name')
args = parser.parse_args()

url_dict = {'bulk_dos': 'https://drive.google.com/uc?id=1mLQYkamCFf-68FE_crt476mShs-zOue1'}

url = url_dict[args.input_file]
output = args.input_file + '.json'

# download file from url to output
gdown.download(url, output, quiet=False)

