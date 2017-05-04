
from Capstone import constants as cnts
from Capstone.one_bit_matrix_completion import OneBitMatrixCompletion
from Capstone.movielense_processor import MovielenseProcessor
from Capstone import functions as fn
import numpy as np

class TopProcessor:
    """
    This class encompasses all the top level functionality for retrieving the data from a source and running the 
    recommendation system. 
    """
    movielense_proc = MovielenseProcessor
    one_bit_completor = OneBitMatrixCompletion

    def __init__(self, df=None):
        """ This method initializes the main parameters using the inputs given
        
        Parameters
        ----------
        :param th: 
        :type th: float
        :param df: The input data set to be processed and analysed
        :type df: str
        :param pd: 
        """
        self.movielense_proc = MovielenseProcessor()
        self.one_bit_completor = OneBitMatrixCompletion()

    def run(self):
        M = self.movielense_proc.extract_rating_matrix('../data/u_data.xlsx', 943, 1682)#np.matrix([[1, cnts.NO_VALUE,
        #  1], [-1, 1, 1], [cnts.NO_VALUE, 1, cnts.NO_VALUE]])
        Mhat = self.one_bit_completor.complete(M)

        #TODO: Do something with Mhat
        d1, d2 = Mhat.shape
        for i in range(d1):
            for j in range(d2):
                val = fn.logistic(Mhat[i, j])
                print(val)
                if val >= 1 / 2:
                    Mhat[i, j] = 1
                else:
                    Mhat[i, j] = -1
        np.save('output', Mhat)

if __name__ == "__main__":
    one_compet = TopProcessor()
    one_compet.run()
