import pandas 
import numpy as np
import h5py


class DataReader:
    """
    Manages data import from .h5 file to pandas.DataFrame
    
    """
    
    def __init__(self, filename: str):
        """
        Reads data from .h5 file

        Args:
            filename (str): full or relative path to the data file
        """
        
        # store the file name
        self.filename: str = filename
        
        # read hdf keys and store them
        f = h5py.File(self.filename, "r")
        self.keys: list = f.keys()
        
        # build the whole dataset 
        df = pandas.concat(
            [
                pandas.read_hdf(self.filename, key=k, mode="r") for k in self.keys
            ],
            ignore_index=True
        )
        
        # store the dataset with new column names
        self.df : pandas.DataFrame = df.rename(columns={"CH":"ch", "HIT_DRIFT_TIME":"drift_time", "THETA":"theta"})
    
    
    
    def build_sample(self, ndata: int) -> pandas.DataFrame:
        """
        Builds the dataset of ndata samples

        Args:
            ndata (int): dimensionality of the dataset

        Returns:
            df (pandas.DataFrame): dataset
        """
        
        # random data extraction from the whole dataset
        df = self.df.sample(n=ndata)
        
        return df[["drift_time", "theta"]] 
    
    
    
    def cut_theta(self, ndata: int = 0, theta1: float = None, theta2: float = None) -> pandas.DataFrame:
        """
        Performs cuts on the theta observable 
        
        Args:
            ndata (int): dimensionality of the dataset
            theta1 (float): lower bound (if not specified theta = theta < theta2)
            theta2 (float): upper bound (if not specigied theta = theta > theta1) 
            
        Returns:
            df (pandas.DataFrame): dataset after theta cut
            
        """
        
        # interval theta1 < theta < theta2
        if theta1 and theta2:
            df = self.df[(np.abs(self.df["theta"])>theta1) & (np.abs(self.df["theta"])<theta2)]
<<<<<<< HEAD
            print(f"{theta1} < theta < {theta2}")
        # interval theta > theta1
        elif theta1 and not theta2:
            df = self.df[np.abs(self.df["theta"])>theta1]
            print(f"theta > {theta1}")
        # interval theta < theta2
        elif not theta1 and theta2:
            df = self.df[np.abs(self.df["theta"])<theta2]
            print(f"theta < {theta2}")
        # no cut
        elif not theta1 and not theta2:
=======
        elif theta1:
            df = self.df[np.abs(self.df["theta"])>theta1]
        elif theta2:
            df = self.df[np.abs(self.df["theta"])<theta2]
        else:
>>>>>>> d7bbfbcd4a7d87798d2587ab98edf5823b19cfe9
            df = self.df
            print("no cut performed")
            
        if ndata:
            df = df.sample(n=ndata)

        return df[["drift_time", "theta"]] 


