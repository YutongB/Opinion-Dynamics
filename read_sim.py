from sim import read_results
from original_compat import *
import argparse

def main(args):

    res = read_results(args.fname)

    

    print(res[0].friendliness)
    print(res[0].initial_distr)
    print(res[0].adjacency)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Opinion dynamics simulation reader, Yutong Bu 2021')

    parser.add_argument("fname", help="filename with results to read")
    
    args = parser.parse_args()

    main(args)