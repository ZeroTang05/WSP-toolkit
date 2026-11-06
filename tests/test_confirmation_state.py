import unittest
from confirmation_state import canonical_key


class ConfirmationStateTests(unittest.TestCase):
    def test_zero_direction_reuses_native_base(self):
        base={'model':'original','tokenizer':'v1'}
        self.assertEqual(canonical_key(base),canonical_key(base,'some-delta',[0.,-0.,0,0]))

    def test_integer_coefficients_have_same_compiled_state(self):
        self.assertEqual(canonical_key({'model':'a'},'d',[1,1,1,1]),canonical_key({'model':'a'},'d'))
        self.assertEqual(canonical_key({'model':'a'},'d',[.5,0,1,0]),canonical_key({'model':'a'},'d',[.5,-0.,1.,0.]))

    def test_distinct_weights_or_nonzero_coefficients_are_not_merged(self):
        original=canonical_key({'model':'a'},'d',[.5,0.,1.,0.])
        self.assertNotEqual(original,canonical_key({'model':'b'},'d',[.5,0.,1.,0.]))
        self.assertNotEqual(original,canonical_key({'model':'a'},'other',[.5,0.,1.,0.]))
        self.assertNotEqual(original,canonical_key({'model':'a'},'d',[.5000000001,0.,1.,0.]))

    def test_invalid_cache_requests_are_rejected(self):
        for point in [[1,1,1],[float('nan')]*4,[2]*4]:
            with self.assertRaises(ValueError):canonical_key({'model':'a'},'d',point)
        with self.assertRaises(ValueError):canonical_key({'model':'a'},None,[1.]*4)
