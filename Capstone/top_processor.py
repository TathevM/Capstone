import numpy

from Capstone import constants as cnts
from Capstone.one_bit_matrix_completion import OneBitMatrixCompletion
from Capstone.movielense_processor import MovielenseProcessor
from Capstone import functions as fn
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl

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
        self.one_bit_completor = OneBitMatrixCompletion(500, 5, -0.5, 0.1)

    def run(self):
        M = self.movielense_proc.extract_rating_matrix('../data/u_data.xlsx', 943, 1682)
        complete_M = self.movielense_proc.extract_rating_matrix('../data/complete_m.xlsx',
                                                                943, 1682)

        # Mhat = self.one_bit_completor.complete(M)
        s = '../data/first5000.xlsx'
        # t, f = self.validate(Mhat, s)
        # print(t, f)

        r_vals = [10, 900, 943]
        step_vals = [5, 10, 50, 100]
        alph_vals = [0.01, 0.1, 0.5, 1]
        gamm_vals = [0.01, 0.1, 0.5, 1]

        errors = []
        for r in r_vals:
            for step in step_vals:
                print('Iteration for (' + str(r) + ', ' + str(step) + ')')
                self.one_bit_completor.r = r
                self.one_bit_completor.num_steps = step
                Mhat = self.one_bit_completor.complete(M)
                error = self.validate(Mhat, s)
                print(error)
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

    def validate(self, mat, data):
        true_val = 0
        false_val = 0
        wb = xl.load_workbook(data)
        sheet = wb.active
        rows = sheet.rows

        for row in rows:
            u_id = row[0].value - 1
            m_id = row[1].value - 1
            rating = row[2].value
            if mat[u_id, m_id] == 1 and rating > 2:
                true_val += 1
                print('KUKU!')
            elif mat[u_id, m_id] == -1 and rating <= 2:
                true_val += 1
            else:
                false_val += 1
        return true_val, false_val

        # print(true_val,false_val)


if __name__ == "__main__":
    one_compet = TopProcessor()
    one_compet.run()
