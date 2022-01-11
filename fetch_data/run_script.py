import os
import argparse
import os.path
import time

"""

Python script that uses condor to exploit the CPU cluster (available only through t2-ui-12 machine via ssh) to distribute jobs to different computing resources.

USAGE:

python run_condor_script.py -o output_data_directory -n number_of_files_to_read -p python_script_file


Example: 

python run_script.py -o ../data/ -n -1 -p download_data.py

CONSIDER SETTING UP MEANINGFUL DEFAULT VALUES SO THAT IT BECOMES

python run_condor_script.py

"""


def argParser():
    """manages command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-o"  , "--output"  , type=str, default="../data/"          , help="output directory")
    parser.add_argument("-n"  , "--nfiles"  , type=int, default=-1                  , help="number of files" )
    parser.add_argument("-p"  , "--pyscript", type=str, default="./download_data.py", help="python script to execute")

    return parser.parse_args()


def main(args):
    """
        MAIN FUNCTION gets data from CloudVeneto and save them into a .txt file
        
        args: the command line arguments  
    """
    
    OUTPUT_PATH   = args.output
    PYTHON_SCRIPT = args.pyscript
    N_FILES       = args.nfiles
    
    runs = [
        1231
    ]
    
    job_folder = "condor"+str(time.time())
    condor_path = "./condor_jobs/"
    job_path = condor_path+job_folder
    if not os.path.exists(condor_path):
        os.makedirs(condor_path)
    os.makedirs(job_path)
    
    for run in runs:
        joblabel=f"RUN{run}"
        
        # src file
        with open(f"{job_path}/{joblabel}.src" , "w") as script_src:
            script_src.write("#!/bin/bash\n")
            script_src.write('eval "$(/lustre/cmswork/nlai/anaconda/bin/conda shell.bash hook)" \n')
            script_src.write(f"python {os.getcwd()}/{PYTHON_SCRIPT} -o {OUTPUT_PATH} -run {run} -n {N_FILES} ") 

        os.system(f"chmod a+x {job_path}/{joblabel}.src") # THIS MAKES THE FILE EXECUTABLE
        
        # condor file
        with open(f"{job_path}/{joblabel}.condor", "w") as script_condor:
            script_condor.write(f"executable = {job_path}/{joblabel}.src\n" )
            script_condor.write("universe = vanilla\n")
            script_condor.write(f"output = {job_path}/{joblabel}.out\n" )
            script_condor.write(f"error =  {job_path}/{joblabel}.err\n" )
            script_condor.write(f"log = {job_path}/{joblabel}.log\n")
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write("queue\n")
            
        # condor file submission
        os.system(f"condor_submit {job_path}/{joblabel}.condor") 
    
    return


if __name__ == "__main__":
    args = argParser()
    main(args)