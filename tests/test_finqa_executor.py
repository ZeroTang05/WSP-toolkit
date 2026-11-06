import unittest
from finqa_executor import execute


class FinQAExecutorTests(unittest.TestCase):
    def test_official_arithmetic_percentage_reference_and_table(self):
        self.assertEqual(execute('divide(20, 25%), add(#0, const_m1)',[])['result'],79.)
        self.assertEqual(execute('table_sum(revenue, none)',[['revenue','1,000','$ 3']])['result'],1003.)
        self.assertEqual(execute('greater(10, 2)',[])['result'],'yes')

    def test_malformed_nonfinite_forward_references_and_python_rejected(self):
        for program in ['add(#0, 2)','divide(1, 0)','add(nan, 1)','add(inf, 1)',
                        'add(1, 2) trailing','__import__(os)','add(1, 2),', '']:
            self.assertFalse(execute(program,[])['valid'],program)
