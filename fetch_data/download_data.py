import argparse
from ReadStream import StreamReader

""" USAGE:

    python download_data.py -o <output_data_directory> -run <last_4_digits_of_run> 
    -n <number_of_files_to_read> -format <upper or lower>

    The output data directory should be ../data/ unless you want to save data somewhere else (not recommended). 
    The ../data/ folder will be created while running this script the first time, and it has already been added to .gitignore!


    Example: to download the old data RUN001231

    python download_data.py -run 1231 -format upper


    Example: to download a small portion of RUN001231 into the ../data/ folder simply type

    python download_data.py -run 1231 -n 20 -format upper
    
    
    IF data is NEW data then set -format lower
    
    Example: to download the new data Run000054
    
    python download_data.py -run 0054 -format lower
    


    If you change the output directory please make sure to add the new folder to .gitignore
    
"""


def argParser():
    """manages command line arguments"""
    
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "-o", "--output", type=str, default="../data/", help="output directory"
    )
    parser.add_argument(
        "-format", "--format", type=str, default="upper", help="CVeneto container format"
    )
    parser.add_argument(
        "-run", "--run", type=str, default="0054", help="run number"
    )
    parser.add_argument(
        "-n", "--nfiles", type=int, default=-1, help="number of files"
    )
    
    return parser.parse_args()


def main(args):
    """
        MAIN FUNCTION gets data from CloudVeneto and save them into a .txt file
        
        args: the command line arguments  
    """

    # store the command line arguments
    RUNNUMBER = args.run
    OUT_PATH  = args.output
    N_FILES   = args.nfiles
    FORMAT    = args.format 
    
    # create an instance of StreamReader to manage data I/O from CloudVeneto
    reader = StreamReader(RUNNUMBER, OUT_PATH, N_FILES, FORMAT)
    
    # get data from CloudVeneto
    reader.readStream()
    # save data into a .txt file
    reader.saveData()
    
    print("\n\nExiting...\n\n")


if __name__ == "__main__":
    args = argParser()
    main(args)
    