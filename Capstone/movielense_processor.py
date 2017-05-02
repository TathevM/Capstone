import openpyxl as xl
import numpy
from Capstone import constants as cnts
from openpyxl import Workbook


class MovielenseProcessor:
    def __init__(self):
        pass

    def extract_rating_matrix(self, data_file, u_cnt, m_cnt):
        wb = xl.load_workbook(data_file)
        sheet = wb.active
        rows = sheet.rows
        rating_matrix = numpy.ones((u_cnt, m_cnt)) * cnts.NO_VALUE

        for row in rows:
            u_id = row[0].value - 1
            m_id = row[1].value - 1
            val = row[2].value
            if val <= 2:
                rating_matrix[u_id, m_id] = -1
            else:
                rating_matrix[u_id, m_id] = 1

        return rating_matrix


if __name__ == "__main__":
    mp = MovielenseProcessor()
    mat = mp.extract_rating_matrix('../data/data_set.xlsx', 943, 1682)
    print(mat)
