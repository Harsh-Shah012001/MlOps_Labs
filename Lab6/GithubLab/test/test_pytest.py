import pytest
from src import temperature

def test_celsius_to_fahrenheit():
    assert temperature.celsius_to_fahrenheit(0) == 32
    assert temperature.celsius_to_fahrenheit(100) == 212
    assert temperature.celsius_to_fahrenheit(-40) == -40
    assert temperature.celsius_to_fahrenheit(37) == 98.6

def test_fahrenheit_to_celsius():
    assert temperature.fahrenheit_to_celsius(32) == 0
    assert temperature.fahrenheit_to_celsius(212) == 100
    assert temperature.fahrenheit_to_celsius(-40) == -40
    assert round(temperature.fahrenheit_to_celsius(98.6), 1) == 37.0

def test_celsius_to_kelvin():
    assert temperature.celsius_to_kelvin(0) == 273.15
    assert temperature.celsius_to_kelvin(100) == 373.15
    assert temperature.celsius_to_kelvin(-273.15) == 0

def test_kelvin_to_celsius():
    assert temperature.kelvin_to_celsius(273.15) == 0
    assert temperature.kelvin_to_celsius(373.15) == 100
    assert temperature.kelvin_to_celsius(0) == -273.15
