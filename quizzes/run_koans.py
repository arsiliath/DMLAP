#!/usr/bin/env python3
"""
Koan Runner
===========

Run your koans and see which one to work on next.

Usage:
    python run_koans.py          Run all koans, show first failure
    python run_koans.py --hint   Show a hint for the current failing koan
    python run_koans.py --all    Run all koans, show all results
"""

import sys
import traceback
import re

# Hints for each koan
HINTS = {
    'koan_01': "Lists start at index 0. What index gives you the first element?",
    'koan_02': "-1 gives you the last element. What about -2?",
    'koan_03': "Slice syntax is [start:end]. To get indices 2 and 3, what values do you need?",
    'koan_04': "The method is called 'append'. What string do you want to add?",
    'koan_05': "To double n, you can write n * 2 or n + n",
    'koan_06': "The key is a string. Don't forget the quotes: 'name'",
    'koan_07': "You need a key and a value: 'language': 'Python'",
    'koan_08': "range(5) gives [0,1,2,3,4]. What number gives [0,1,2,3,4]?",
    'koan_09': "The keyword is 'in'. Like: x in my_list",
    'koan_10': "You need a variable name on the right side of the comma. Convention is 'y'.",
    'koan_11': "The standard alias for numpy is 'np'",
    'koan_12': "Shape is (rows, cols). For 2 rows and 3 columns: (2, 3)",
    'koan_13': "arange(6) gives [0, 1, 2, 3, 4, 5]. What's 6?",
    'koan_14': "3 values from 0 to 1: that's [0, 0.5, 1]. How many values?",
    'koan_15': "The attribute is called 'shape' (no parentheses)",
    'koan_16': "For transpose, use the attribute 'T' (capital letter, no parentheses)",
    'koan_17': "Matrix multiplication uses the @ operator",
    'koan_18': "To reverse, use slice [::-1] - that's start:end:step with step=-1",
    'koan_19': "The function is 'seed'. np.random.seed(42)",
    'koan_20': "argmax() returns the index of the maximum. It's a method.",
    'koan_21': "The standard alias for matplotlib.pyplot is 'plt'",
    'koan_22': "The function to show images is 'imshow' (im + show)",
    'koan_23': "The colormap parameter is called 'cmap'",
    'koan_24': "First argument is rows, second is columns: subplots(2, 3)",
    'koan_25': "The figure size parameter is 'figsize'",
}


def get_koan_number(func_name):
    """Extract koan number from function name."""
    match = re.match(r'koan_(\d+)', func_name)
    return int(match.group(1)) if match else 0


def run_koan(koan_func):
    """Run a single koan and return (success, error_message)."""
    try:
        koan_func()
        return True, None
    except AssertionError as e:
        return False, str(e)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except NameError as e:
        return False, f"Name error: {e}"
    except TypeError as e:
        return False, f"Type error: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def print_mountain():
    """Print ASCII art."""
    print("""
                    _______
                   /       \\
                  /  PYTHON \\
                 /   KOANS   \\
                /     ___     \\
               /     /   \\     \\
              /     /     \\     \\
             /     /       \\     \\
            /_____/         \\_____\\
    """)


def main():
    show_hint = '--hint' in sys.argv
    show_all = '--all' in sys.argv

    # Import koans
    try:
        from python_koans import ALL_KOANS
    except ImportError:
        # Try relative import for different working directories
        try:
            import importlib.util
            import os
            koan_path = os.path.join(os.path.dirname(__file__), '01_python_koans.py')
            spec = importlib.util.spec_from_file_location("python_koans", koan_path)
            koans_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(koans_module)
            ALL_KOANS = koans_module.ALL_KOANS
        except Exception as e:
            print(f"Error loading koans: {e}")
            print("Make sure you're running from the quizzes directory.")
            sys.exit(1)

    print_mountain()

    passed = 0
    failed = 0
    first_failure = None

    print("=" * 60)
    print("  Running Python Koans...")
    print("=" * 60 + "\n")

    for koan in ALL_KOANS:
        success, error = run_koan(koan)
        koan_name = koan.__name__
        koan_num = get_koan_number(koan_name)

        if success:
            passed += 1
            if show_all:
                print(f"  [PASS] {koan_name}")
        else:
            failed += 1
            if first_failure is None:
                first_failure = (koan_name, koan.__doc__, error, koan_num)
            if show_all:
                print(f"  [FAIL] {koan_name}")

    print()
    print("=" * 60)

    # Progress bar
    total = len(ALL_KOANS)
    bar_width = 40
    filled = int(bar_width * passed / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  Progress: [{bar}] {passed}/{total}")
    print("=" * 60)

    if first_failure:
        name, doc, error, num = first_failure
        print(f"\n  Your journey continues with: {name}")
        print(f"  {doc}")
        print(f"\n  The error:")
        print(f"    {error}")

        if show_hint:
            hint_key = f"koan_{num:02d}"
            hint = HINTS.get(hint_key, "No hint available.")
            print(f"\n  HINT: {hint}")
        else:
            print(f"\n  Need help? Run: python run_koans.py --hint")

        print(f"\n  Edit 01_python_koans.py and replace the __ with your answer.")
        print()
    else:
        print("""
  ============================================================

    CONGRATULATIONS!

    You have completed all the Python koans!

    You have learned:
      - Python lists, dicts, and control flow
      - NumPy array creation and operations
      - Matplotlib basics

    You are ready to continue your machine learning journey.

  ============================================================
        """)


if __name__ == '__main__':
    main()
