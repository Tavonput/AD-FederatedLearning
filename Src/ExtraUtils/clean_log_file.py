import re


def main() -> None:
    """ Filter out the lines that contain either of the following:

    - [DRIVER] Stopping evaluators
    - [EVAL 0] Stopping
    - FederatedResults

    This was used for the 'collapse analysis test' experiment to determine which configurations were overloading the
    evaluator.   
    """
    pattern = re.compile(r'\[DRIVER\] Stopping evaluators|\[EVAL 0\] Stopping|FederatedResults')

    with open("log.log", "r") as infile, open("cleaned_log.log", "w") as outfile:
        for line in infile:
            if pattern.search(line):
                outfile.write(line)


if __name__ == "__main__":
    main()
