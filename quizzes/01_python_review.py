"""
Introduction to Python - Review
================================

Fill in the blanks to make the tests pass.

Run:  python run_review.py
Hint: python run_review.py --hint
"""

# =============================================================================
# This is your placeholder. Replace it with the correct value.
# =============================================================================
FILL_ME_IN = "???"


# =============================================================================
# PYTHON BASICS - Lists
# =============================================================================

def q01_list_indexing():
    """Access elements in a list by their index."""
    fruits = ['apple', 'banana', 'cherry', 'date']

    # Lists are zero-indexed. Replace FILL_ME_IN with the correct index.
    first_fruit = fruits[FILL_ME_IN]  # Should get 'apple'

    assert first_fruit == 'apple', f"Expected 'apple', got {first_fruit!r}"


def q02_negative_indexing():
    """Access elements from the end using negative indices."""
    numbers = [10, 20, 30, 40, 50]

    # Negative indices count from the end. -1 is the last element.
    last_number = numbers[FILL_ME_IN]  # Should get 50

    assert last_number == 50, f"Expected 50, got {last_number}"


def q03_list_slicing():
    """Extract a portion of a list using slicing."""
    letters = ['a', 'b', 'c', 'd', 'e', 'f']

    # Slicing: [start:end] - end is exclusive
    # Replace the two FILL_ME_IN values with numbers
    middle = letters[FILL_ME_IN:FILL_ME_IN]  # Should get ['c', 'd']

    assert middle == ['c', 'd'], f"Expected ['c', 'd'], got {middle}"


def q04_list_append():
    """Add an element to the end of a list."""
    colors = ['red', 'green']

    # Which method adds an element to the end of a list?
    # Replace FILL_ME_IN with the method name (as a string)
    method_name = FILL_ME_IN  # 'append', 'insert', 'extend', or 'add'?

    getattr(colors, method_name)('blue')

    assert colors == ['red', 'green', 'blue'], f"Expected ['red', 'green', 'blue'], got {colors}"


def q05_list_comprehension():
    """Create a new list by transforming each element."""
    numbers = [1, 2, 3, 4, 5]

    # List comprehension: [expression for item in list]
    # Replace FILL_ME_IN with an expression that doubles n
    doubled = [FILL_ME_IN for n in numbers]

    assert doubled == [2, 4, 6, 8, 10], f"Expected [2, 4, 6, 8, 10], got {doubled}"


# =============================================================================
# PYTHON BASICS - Dictionaries
# =============================================================================

def q06_dict_access():
    """Access values in a dictionary by key."""
    person = {'name': 'Alice', 'age': 30, 'city': 'London'}

    # Use the key to get the value
    name = person[FILL_ME_IN]  # Should get 'Alice'

    assert name == 'Alice', f"Expected 'Alice', got {name!r}"


def q07_dict_creation():
    """Create a dictionary with key-value pairs."""

    # Create a dict where 'language' maps to 'Python'
    # Replace both FILL_ME_IN values
    data = {FILL_ME_IN: FILL_ME_IN}

    assert data.get('language') == 'Python', f"Expected {{'language': 'Python'}}, got {data}"


# =============================================================================
# PYTHON BASICS - Control Flow
# =============================================================================

def q08_range():
    """Understand how range() generates sequences."""

    # range(n) generates 0, 1, 2, ... n-1
    # What value of n gives [0, 1, 2, 3, 4]?
    result = list(range(FILL_ME_IN))

    assert result == [0, 1, 2, 3, 4], f"Expected [0, 1, 2, 3, 4], got {result}"


def q09_membership():
    """Check if an element exists in a collection."""
    vowels = ['a', 'e', 'i', 'o', 'u']

    # What keyword checks if something is in a collection?
    # Answer should be: 'in' or 'not in'
    keyword = FILL_ME_IN

    is_vowel = eval(f"'e' {keyword} vowels")

    assert is_vowel == True, f"Expected True. Make sure keyword is 'in'"


def q10_tuple_unpacking():
    """Unpack a tuple into multiple variables."""
    coordinates = (10, 20)

    # Tuple unpacking assigns each element to a variable
    # What expression unpacks the tuple into x and y?
    # Hint: the answer looks like "x, y"
    unpack_expression = FILL_ME_IN

    exec(f"{unpack_expression} = coordinates")
    result = eval(unpack_expression.replace(',', '+'))  # Adds x + y

    assert result == 30, f"Expected x=10 and y=20 (sum=30). Did you use 'x, y'?"


# =============================================================================
# NUMPY - Array Creation
# =============================================================================

def q11_numpy_import():
    """Import numpy with its conventional alias."""

    # What is the standard alias for numpy?
    alias = FILL_ME_IN  # Two letters...

    exec(f"import numpy as {alias}", globals())

    assert alias == 'np', f"The conventional numpy alias is 'np', not '{alias}'"


def q12_zeros_array():
    """Create an array filled with zeros."""
    import numpy as np

    # Create a 2x3 array of zeros (2 rows, 3 columns)
    # Replace with the correct shape tuple
    shape = FILL_ME_IN  # e.g., (rows, cols)

    arr = np.zeros(shape)

    assert arr.shape == (2, 3), f"Expected shape (2, 3), got {arr.shape}"


def q13_arange():
    """Create an array with a range of values."""
    import numpy as np

    # arange(n) creates [0, 1, 2, ..., n-1]
    # What value creates [0, 1, 2, 3, 4, 5]?
    n = FILL_ME_IN

    arr = np.arange(n)

    assert list(arr) == [0, 1, 2, 3, 4, 5], f"Expected [0, 1, 2, 3, 4, 5], got {list(arr)}"


def q14_linspace():
    """Create evenly spaced values over an interval."""
    import numpy as np

    # linspace(start, stop, num) includes both endpoints
    # How many values gives [0.0, 0.5, 1.0]?
    num_values = FILL_ME_IN

    arr = np.linspace(0, 1, num_values)

    expected = [0.0, 0.5, 1.0]
    assert list(arr) == expected, f"Expected {expected}, got {list(arr)}"


# =============================================================================
# NUMPY - Array Properties
# =============================================================================

def q15_array_shape():
    """Access the dimensions of an array."""
    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # What attribute gives the array's dimensions?
    attr_name = FILL_ME_IN  # A 5-letter word...

    dims = getattr(arr, attr_name)

    assert dims == (2, 3), f"Expected (2, 3), got {dims}"


def q16_array_transpose():
    """Swap rows and columns of a matrix."""
    import numpy as np

    matrix = np.array([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2)

    # What attribute gives the transpose?
    attr_name = FILL_ME_IN  # Just one letter...

    transposed = getattr(matrix, attr_name)

    assert transposed.shape == (2, 3), f"Expected shape (2, 3), got {transposed.shape}"


# =============================================================================
# NUMPY - Array Operations
# =============================================================================

def q17_matrix_multiply():
    """Perform matrix multiplication."""
    import numpy as np

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    # What operator performs matrix multiplication?
    # Answer should be: '@' or '*'
    operator = FILL_ME_IN

    result = eval(f"a {operator} b")

    expected = np.array([[19, 22], [43, 50]])
    assert (result == expected).all(), f"Expected matrix product, got {result}"


def q18_array_flip():
    """Reverse an array along an axis."""
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5])

    # What slice reverses an array?
    # Hint: it's start:end:step with step being negative
    slice_str = FILL_ME_IN  # Something like '::-1'

    reversed_arr = eval(f"arr[{slice_str}]")

    assert list(reversed_arr) == [5, 4, 3, 2, 1], f"Expected [5, 4, 3, 2, 1], got {list(reversed_arr)}"


def q19_random_seed():
    """Set the random seed for reproducibility."""
    import numpy as np

    # What function sets the random state?
    func_name = FILL_ME_IN  # A 4-letter word...

    getattr(np.random, func_name)(42)
    first = np.random.rand()

    np.random.seed(42)
    second = np.random.rand()

    assert first == second, f"Random seed not set. The function is 'seed'."


def q20_argmax():
    """Find the index of the maximum value."""
    import numpy as np

    arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

    # What method returns the INDEX of the max value?
    method_name = FILL_ME_IN  # arg + max = ?

    max_index = getattr(arr, method_name)()

    assert max_index == 5, f"Expected 5 (index of 9), got {max_index}"


# =============================================================================
# MATPLOTLIB
# =============================================================================

def q21_plot_import():
    """Import matplotlib.pyplot with its conventional alias."""

    # What is the standard alias for matplotlib.pyplot?
    alias = FILL_ME_IN  # Three letters...

    assert alias == 'plt', f"The conventional alias is 'plt', not '{alias}'"


def q22_imshow_function():
    """Display an image using the correct function."""

    # What function displays images in matplotlib?
    func_name = FILL_ME_IN  # im + show = ?

    assert func_name == 'imshow', f"Expected 'imshow', got '{func_name}'"


def q23_grayscale_cmap():
    """Display a grayscale image without false colors."""

    # What parameter name specifies the colormap?
    # plt.imshow(img, ????='gray')
    param_name = FILL_ME_IN  # Short for "colormap"

    assert param_name == 'cmap', f"Expected 'cmap', got '{param_name}'"


def q24_subplots():
    """Create a grid of subplots."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Create a figure with 2 rows and 3 columns
    # What tuple creates this grid?
    grid = FILL_ME_IN  # (rows, cols)

    fig, axes = plt.subplots(*grid)

    assert axes.shape == (2, 3), f"Expected shape (2, 3), got {axes.shape}"
    plt.close(fig)


def q25_figsize():
    """Create a figure with a specific size."""

    # What parameter name specifies figure dimensions?
    # plt.figure(????=(10, 5))
    param_name = FILL_ME_IN  # fig + size = ?

    assert param_name == 'figsize', f"Expected 'figsize', got '{param_name}'"


# =============================================================================
# All questions
# =============================================================================

ALL_QUESTIONS = [
    q01_list_indexing,
    q02_negative_indexing,
    q03_list_slicing,
    q04_list_append,
    q05_list_comprehension,
    q06_dict_access,
    q07_dict_creation,
    q08_range,
    q09_membership,
    q10_tuple_unpacking,
    q11_numpy_import,
    q12_zeros_array,
    q13_arange,
    q14_linspace,
    q15_array_shape,
    q16_array_transpose,
    q17_matrix_multiply,
    q18_array_flip,
    q19_random_seed,
    q20_argmax,
    q21_plot_import,
    q22_imshow_function,
    q23_grayscale_cmap,
    q24_subplots,
    q25_figsize,
]
