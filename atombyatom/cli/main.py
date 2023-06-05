# cli/main.py
import argparse
from .download import CLICommand

def main():
    # Create a top-level parser
    parser = argparse.ArgumentParser(prog='atombyatom')

    subparsers = parser.add_subparsers(dest='command')

    # Create a parser for the 'download' command
    download_parser = subparsers.add_parser('download', help='download data')
    download_parser.add_argument('dataset', help='Dataset to download')

    args = parser.parse_args()

    # Initialize the CLICommand class
    command = CLICommand()

    # Based on the command name, call the appropriate function
    if args.command == 'download':
        command.run(args.dataset)

if __name__ == "__main__":
    main()
