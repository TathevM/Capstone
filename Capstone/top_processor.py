
from Capstone import constants as cnts
from Capstone.one_bit_matrix_completion import OneBitMatrixCompletion
from Capstone.movielense_processor import MovielenseProcessor
from Capstone import functions as fn
import numpy as np
import matplotlib.pyplot as plt

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
        self.one_bit_completor = OneBitMatrixCompletion(10, 5, 0.5, 0.1)

    def run(self):
        M = self.movielense_proc.extract_rating_matrix('../data/test.xlsx', 10, 15)
        complete_M = self.movielense_proc.extract_rating_matrix('../data/small_data.xlsx',
                                                                10, 15)

        r_vals = [2, 5, 10]
        step_vals = [5, 100, 1000, 10000]
        alph_vals = [0.01, 0.1, 0.5, 1]
        gamm_vals = [0.01, 0.1, 0.5, 1]

        errors = []
        for r in r_vals:
            for step in step_vals:
                self.one_bit_completor.r = r
                self.one_bit_completor.num_steps = step
                Mhat = self.one_bit_completor.complete(M)
                error = self.compute_least_square_error(complete_M, Mhat)
                errors.append(error)

        print(errors)
        # self.plot_errors(r_vals, step_vals, errors)
        np.save('output', Mhat)

    def compute_least_square_error(self, M, Mhat):
        """

        :param M: 
        :param Mhat: 
        """
        diff = Mhat - M
        max_diff = np.max(diff)
        if max_diff == 0:
            return 0

        diff_norm = np.divide(diff, max_diff)
        lst = np.sqrt(np.sum(diff_norm ** 2))
        return lst

    def plot_errors(self, r_vals, step_vals, errors):
        pass


if __name__ == "__main__":
    one_compet = TopProcessor()
    one_compet.run()
