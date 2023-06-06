# cli/main.py
import argparse
from .download import CLICommand
from .run import CLICommand

def main():
    # Create a top-level parser
    parser = argparse.ArgumentParser(prog='atombyatom')

    subparsers = parser.add_subparsers(dest='command')

    # Create a parser for the 'download' command
    download_parser = subparsers.add_parser('download', help='download data')
    download_parser.add_argument('dataset', help='Dataset to download')

    # Create a parser for the 'run' command
    run_parser = subparsers.add_parser('run', help='run model')
    run_parser.add_argument('model', help='Model to run')
    run_parser.add_argument('--dataset', type=str, default='bulk_dos', help='Dataset to run on')

    args = parser.parse_args()

    # Initialize the CLICommand class
    command = CLICommand()

    # Based on the command name, call the appropriate function
    if args.command == 'download':
        command.run(args.dataset)

    elif args.command == 'run':
        command.run(args.model, args.dataset)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
