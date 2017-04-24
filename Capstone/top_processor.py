
from Capstone import constants as cnts
from Capstone.one_bit_matrix_completion import OneBitMatrixCompletion
from Capstone.movielense_processor import MovielenseProcessor

class TopProcessor:
    """
    This class encompasses all the top level functionality for retrieving the data from a source and running the 
    recommendation system. 
    """
    movielense_proc = MovielenseProcessor
    one_bit_completor = OneBitMatrixCompletion

    def __init__(self, df=None, th=None, pd=cnts.LOGISTIC_FUNCTION):
        """ This method initializes the main parameters using the inputs given
        
        Parameters
        ----------
        :param th: 
        :type th: float
        :param df: The input data set to be processed and analysed
        :type df: str
        :param pd: 
        """
        self.movielense_proc = MovielenseProcessor(data_file=df)
        self.one_bit_completor = OneBitMatrixCompletion(th, pd)

    def run(self):
        M = self.movielense_proc.extract_rating_matrix()
        Mhat = self.one_bit_completor.complete(M)
        #TODO: Do something with Mhat

if __name__ == "__main__":
    one_compet = TopProcessor()
    one_compet.run()
