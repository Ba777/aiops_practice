import unittest

import numpy as np
import pandas as pd

from dta_lstm import DTA_LSTM
from config import DEFAULTS


class Test_DTA_LSTM(unittest.TestCase):

    def setUp(self):
        df_traces = pd.DataFrame([['ID01', [1]],
                                  ['ID02', [1, 2]],
                                  ['ID03', [1, 2, 2]],
                                  ['ID04', [1, 2]]],
                                 columns=['traceId', DEFAULTS['span_list_col_name']]).set_index('traceId')

        self.df_train, self.df_test = (df_traces.iloc[0:len(df_traces) // 2],
                                       df_traces.iloc[len(df_traces) // 2:])

    def _invoke_algorithm(self, train, test):
        tm = DTA_LSTM()
        tm.fit(train)
        tm.predict(test)
        return tm

    def test_shift_left(self):
        s = DTA_LSTM()

        a = [([], []),
             ([0], [0]),
             ([1], [0]),
             ([1, 2, 3, 4], [2, 3, 4, 0]),
             ([0, 1, 2, 3, 4], [0, 2, 3, 4, 0]),
             ([0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 2, 3, 4, 0])]

        for _a in a:
            np.testing.assert_array_equal(s._shift_traces_left(np.array([_a[0]])), np.array([_a[1]]))

    def test_normal_execution(self):
        tm = self._invoke_algorithm(train=self.df_train, test=self.df_test)
        df = tm.predict_proba()
        self.assertGreaterEqual(len(df[df != 1.0]), 1)

    def test_wrong_datatype(self):
        with self.assertRaises(TypeError):
            self._invoke_algorithm(train=[], test=[])

