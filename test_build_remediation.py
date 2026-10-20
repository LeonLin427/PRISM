import unittest

from build_remediation import api_deltas, is_generated_component_stub


class RemediationFeatureTests(unittest.TestCase):
    def test_api_delta_uses_side_max_not_sum(self):
        problem = [
            {'api_calls': {'getDeviceId': 4}},
            {'api_calls': {'getDeviceId': 3}},
        ]
        fixed = [
            {'api_calls': {'getDeviceId': 2}},
            {'api_calls': {'getDeviceId': 1}},
        ]
        self.assertEqual(api_deltas(problem, fixed, ['getDeviceId']),
                         {'api_getDeviceId': 2})

    def test_generated_stub_requires_both_random_patterns(self):
        self.assertTrue(is_generated_component_stub(
            'com.example.abcde12.Fghij34Activity'))
        self.assertFalse(is_generated_component_stub(
            'com.example.feature.Fghij34Activity'))
        self.assertFalse(is_generated_component_stub(
            'com.example.abcde12.LoginActivity'))


if __name__ == '__main__':
    unittest.main()
