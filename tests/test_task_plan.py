import unittest
from task_plan import compile_coefficients


class TaskPlanTests(unittest.TestCase):
    def test_normalized_depth_and_invalid_target(self):
        names=[f'model.layers.{i}.self_attn.q_proj' for i in range(28)]
        values=compile_coefficients(names,28,[0,.5,1,1.5])
        self.assertEqual([values[names[i]] for i in [0,7,14,21,27]],[0.,.5,1.,1.5,1.5])
        with self.assertRaises(ValueError):compile_coefficients(['model.layers.30.self_attn.q_proj'],28,[1]*4)
        with self.assertRaises(ValueError):compile_coefficients(names,28,[float('nan')]*4)
