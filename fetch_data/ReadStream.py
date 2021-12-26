from s3fs.core import S3FileSystem
import pandas as pd
import os
import time


class StreamReader:
    """manages data I/O from CloudVeneto"""

    def __init__(self, run_number: int, output_path: str, n_files: int):
        """
        run_number  : idientifies the detector's run - only the last 4 digits are needed
        output_path : directory where data is to be saved
        n_files     : number of files to read from CloudVeneto (usually -1 to get the whole run)
        """

        # initialize instance attributes
        self.run_number = run_number
        self.output_path = output_path
        self.n_files = n_files

        # generate the data file path and name
        self.output_file = self.output_path + f"RUN00{self.run_number}_data.txt"

    def readStream(self):
        """reads data from CloudVeneto"""

        # connection to CVeneto
        print("\nConnecting to CloudVeneto...")
        try:
            start_conn_time = time.process_time()
            s3 = S3FileSystem(
                anon=False,
                key="69a290784f914f67afa14a5b4cadce79",
                secret="2357420fac4f47d5b41d7cdeb52eb184",
                client_kwargs={
                    "endpoint_url": "https://cloud-areapd.pd.infn.it:5210",
                    "verify": False,
                },
            )
        except:
            print("\n\nERROR:")
            print("Unable to establish connection with CloudVeneto")
        else:
            conn_time = time.process_time() - start_conn_time
            print(
                f"Connection with CloudVeneto established correctly in {conn_time:.2f} seconds"
            )

        # read data files from CloudVeneto
        print("\nReading data files...")
        start_read_time = time.process_time()
        # concatenate all data files into a single dataframe
        self.stream_df = pd.concat(
            [
                pd.read_csv(s3.open(f, mode="rb"), encoding="utf8", engine="python")
                for f in s3.ls("/RUN00" + str(self.run_number) + "/")[: self.n_files]
                if f.endswith(".txt")
            ],
            ignore_index=True,
        )

        read_time = time.process_time() - start_read_time
        # feedback
        files_read = "All" if self.n_files == -1 else str(self.n_files)
        print(f"{files_read} data files collected in {read_time:.2f} seconds")

    def saveData(self):
        """saves data into a local file"""

        # if the output directory does not exists
        if not os.path.exists(self.output_path):
            # create the directory before saving data
            print(f"\nCreating {self.output_path} directory")
            os.makedirs(self.output_path)

        print(f"\nSaving data to RUN00{self.run_number}_data.txt...")
        start_save_time = time.process_time()
        self.stream_df.to_csv(self.output_file, index=False, header=True)
        save_time = time.process_time() - start_save_time
        print(f"Saving completed in {save_time:.2f}")
