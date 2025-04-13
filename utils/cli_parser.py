import sys

import sys

def print_usage_and_exit(script_type):
    is_main = script_type == "main"
    print(f"Usage: python {script_type}.py [MPyC options] <dataset.csv>", end=" ")
    if is_main:
        print("[--regression-type|--r] [linear|logistic]", end=" ")
    print("[--normalizer|--n] [minmax|zscore] [--help|-h]")

    print("\nArguments:")
    print("  [MPyC options]     : Optional, like -M (number of parties) or -I (party id)")
    print("  <dataset.csv>      : Path to the local party's CSV file")
    print("  --normalizer -n    : Choose normalization method: 'minmax' or 'zscore', default to none")
    if is_main:
        print("  --regression -r    : Choose regression method: 'linear' or 'logistic', default to 'linear'")
    print("  --help -h          : Show this help message and exit")

    print("\nExample:")
    for i in range(3):
        example = f"  python {script_type}.py -M3 -I{i} party{i}_data.csv -n zscore"
        if is_main:
            example += " -r logistic"
        print(example)
    
    print()
    sys.exit(1)

def parse_cli_args(type):
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage_and_exit(type)

    if len(sys.argv) < 2:
        print_usage_and_exit(type)

    csv_file = None
    normalizer_type = None
    regression_type = "linear"

    # Extract CSV file
    for arg in sys.argv[1:]:
        if not arg.startswith("-") and csv_file is None:
            csv_file = arg

    if csv_file is None:
        print("❌ CSV file not provided.\n")
        print_usage_and_exit()

    # Parse normalizer
    if '--normalizer' in sys.argv:
        idx = sys.argv.index('--normalizer')
        if idx + 1 < len(sys.argv):
            normalizer_type = sys.argv[idx + 1]
    elif '-n' in sys.argv:
        idx = sys.argv.index('-n')
        if idx + 1 < len(sys.argv):
            normalizer_type = sys.argv[idx + 1]

    # Parse regression type
    if '--regression-type' in sys.argv:
        idx = sys.argv.index('--regression-type')
        if idx + 1 < len(sys.argv):
            regression_type = sys.argv[idx + 1]
    elif '-r' in sys.argv:
        idx = sys.argv.index('-r')
        if idx + 1 < len(sys.argv):
            regression_type = sys.argv[idx + 1]

    return {
        "csv_file": csv_file,
        "normalizer_type": normalizer_type,
        "regression_type": regression_type
    }
