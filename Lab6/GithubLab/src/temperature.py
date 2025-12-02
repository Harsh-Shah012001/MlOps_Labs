def celsius_to_fahrenheit(celsius):
    """
    Converts Celsius to Fahrenheit.
    Args:
        celsius (int/float): Temperature in Celsius.
    Returns:
        float: Temperature in Fahrenheit.
    Raises:
        ValueError: If celsius is not a number.
    """
    if not isinstance(celsius, (int, float)):
        raise ValueError("Input must be a number.")
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """
    Converts Fahrenheit to Celsius.
    Args:
        fahrenheit (int/float): Temperature in Fahrenheit.
    Returns:
        float: Temperature in Celsius.
    Raises:
        ValueError: If fahrenheit is not a number.
    """
    if not isinstance(fahrenheit, (int, float)):
        raise ValueError("Input must be a number.")
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    """
    Converts Celsius to Kelvin.
    Args:
        celsius (int/float): Temperature in Celsius.
    Returns:
        float: Temperature in Kelvin.
    Raises:
        ValueError: If celsius is not a number.
    """
    if not isinstance(celsius, (int, float)):
        raise ValueError("Input must be a number.")
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    """
    Converts Kelvin to Celsius.
    Args:
        kelvin (int/float): Temperature in Kelvin.
    Returns:
        float: Temperature in Celsius.
    Raises:
        ValueError: If kelvin is not a number or below absolute zero.
    """
    if not isinstance(kelvin, (int, float)):
        raise ValueError("Input must be a number.")
    if kelvin < 0:
        raise ValueError("Kelvin cannot be negative.")
    return kelvin - 273.15
