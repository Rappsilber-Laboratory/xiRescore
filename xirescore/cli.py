from xirescore.XiRescore import XiRescore

import argparse
import yaml
import ast
import logging
import logging_loki


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Rescoring crosslinked-peptide identifications.')

    # Define CLI arguments
    parser.add_argument('-i', action='store', dest='input_path', help='input path',
                        type=str, required=True)
    parser.add_argument('-o', action='store', dest='output_path', help='output path',
                        type=str, required=False)
    parser.add_argument('-c', action='store', dest='config_file', help='config file',
                        type=str, required=False)
    parser.add_argument('-C', action='store', dest='config_string', help='config test',
                        type=str, required=False)
    parser.add_argument('--loki', action='store', dest='loki', help='Loki server address',
                        type=str, required=False)
    parser.add_argument('--loki-job-id', action='store', dest='loki_job_id', help='Loki job ID',
                        default="", type=str, required=False)

    # Parse arguments
    args = parser.parse_args()

    # Load config
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            options = yaml.safe_load(file)
    elif args.config_string is not None:
        options = ast.literal_eval(args.config_string)
    else:
        options = dict()

    logger = logging.getLogger('xirescore')
    if args.loki is None:
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)
    else:
        handler = logging_loki.LokiHandler(
            url=args.loki,
            tags={
                "application": "xirescore",
                "job_id": args.loki_job_id
            },
            version="1",
        )
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    # Initialize XiRescore
    rescorer = XiRescore(
        input_path=args.input_path,
        output_path=args.output_path,
        options=options,
        logger=logger,
    )
    try:
        rescorer.run()
    except Exception as e:
        logger.fatal(f'Uncaught exception: {e}')


if __name__ == "__main__":
    main()
