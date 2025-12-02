import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import temperature


class TestTemperature(unittest.TestCase):

    def test_celsius_to_fahrenheit(self):
        self.assertEqual(temperature.celsius_to_fahrenheit(0), 32)
        self.assertEqual(temperature.celsius_to_fahrenheit(100), 212)
        self.assertEqual(temperature.celsius_to_fahrenheit(-40), -40)
        self.assertEqual(temperature.celsius_to_fahrenheit(37), 98.6)

    def test_fahrenheit_to_celsius(self):
        self.assertEqual(temperature.fahrenheit_to_celsius(32), 0)
        self.assertEqual(temperature.fahrenheit_to_celsius(212), 100)
        self.assertEqual(temperature.fahrenheit_to_celsius(-40), -40)
        self.assertAlmostEqual(temperature.fahrenheit_to_celsius(98.6), 37.0, places=1)

    def test_celsius_to_kelvin(self):
        self.assertEqual(temperature.celsius_to_kelvin(0), 273.15)
        self.assertEqual(temperature.celsius_to_kelvin(100), 373.15)
        self.assertEqual(temperature.celsius_to_kelvin(-273.15), 0)

    def test_kelvin_to_celsius(self):
        self.assertEqual(temperature.kelvin_to_celsius(273.15), 0)
        self.assertEqual(temperature.kelvin_to_celsius(373.15), 100)
        self.assertEqual(temperature.kelvin_to_celsius(0), -273.15)


if __name__ == '__main__':
    unittest.main()
