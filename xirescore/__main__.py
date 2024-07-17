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
    parser.add_argument('-d', action='store', dest='delimiter', help='CSV delimiter',
                        default=",", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Load input data
    input_file_path = args.input_file
    delimiter = args.delimiter

    if input_file_path.endswith('.csv') or input_file_path.endswith('.csv.gz'):
        input_df = pd.read_csv(input_file_path, delimiter=delimiter)
    elif input_file_path.endswith('.tsv') or input_file_path.endswith('.tsv.gz'):
        input_df = pd.read_csv(input_file_path, delimiter='\t')
    elif input_file_path.endswith('.tab') or input_file_path.endswith('.tab.gz'):
        input_df = pd.read_csv(input_file_path, delimiter='\t')
    elif input_file_path.endswith('.parquet'):
        input_df = pd.read_parquet(input_file_path)

    # Load config
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            options = yaml.safe_load(file)
    else:
        options = dict()

    # Initialize XiRescore
    rescorer = XiRescore(
        input_df=input_df,
        options=options,
    )

    # Run rescorer
    results = rescorer.run()

    # Store results
    output_file_path = args.output_file

    if output_file_path.endswidth('.parquet'):
        results.to_parquet(output_file_path)
    elif output_file_path.endswidth('.csv') or output_file_path.endswidth('.csv.gz'):
        results.to_csv(output_file_path, delimiter=delimiter)
    elif output_file_path.endswidth('.tab') or output_file_path.endswidth('.tab.gz'):
        results.to_csv(output_file_path, delimiter='\t')
    elif output_file_path.endswidth('.tsv') or output_file_path.endswidth('.tsv.gz'):
        results.to_csv(output_file_path, delimiter='\t')


if __name__ == "__main__":
    main()
