from XiRescore import XiRescore

import argparse
import pandas as pd
import yaml


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Rescoring crosslinked-peptide identifications.')

    # Define CLI arguments
    parser.add_argument('-i', action='store', dest='input_file', help='input file',
                        default="", type=str, required=True)
    parser.add_argument('-o', action='store', dest='output_file', help='output file',
                        default="results.csv", type=str, required=True)
    parser.add_argument('-c', action='store', dest='config_file', help='config file',
                        default="", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Load input data
    input_path = args.input_file
    delimiter = args.delimiter

    # Load config
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            options = yaml.safe_load(file)
    else:
        options = dict()

    # Initialize XiRescore
    rescorer = XiRescore(
        input_df=input_path,
        options=options,
    )
    rescorer.run()


if __name__ == "__main__":
    main()
