#!/usr/bin/env python3
"""
Review Runner
=============

Usage:
    python run_review.py          Run questions, show first failure
    python run_review.py --hint   Get a hint for current question
    python run_review.py --all    Show all results
"""

import sys
import re

HINTS = {
    'q01': "Lists start at index 0. What index gives you the first element?",
    'q02': "-1 gives you the last element.",
    'q03': "Slice syntax is [start:end]. To get indices 2 and 3, what values do you need?",
    'q04': "The method is called 'append'.",
    'q05': "To double n, you can write n * 2",
    'q06': "The key is a string. Don't forget the quotes: 'name'",
    'q07': "You need a key and a value: 'language': 'Python'",
    'q08': "range(5) gives [0,1,2,3,4].",
    'q09': "The keyword is 'in'. Like: x in my_list",
    'q10': "You need a variable name for the second value. Try 'x, y'",
    'q11': "The standard alias for numpy is 'np'",
    'q12': "Shape is (rows, cols). For 2 rows and 3 columns: (2, 3)",
    'q13': "arange(6) gives [0, 1, 2, 3, 4, 5].",
    'q14': "3 values from 0 to 1 means [0, 0.5, 1]. Count them.",
    'q15': "The attribute is called 'shape' (no parentheses)",
    'q16': "For transpose, use 'T' (capital letter, no parentheses)",
    'q17': "Matrix multiplication uses the @ operator",
    'q18': "To reverse, use slice [::-1]",
    'q19': "The function is 'seed'. np.random.seed(42)",
    'q20': "argmax() returns the index of the maximum.",
    'q21': "The standard alias for matplotlib.pyplot is 'plt'",
    'q22': "The function to show images is 'imshow'",
    'q23': "The colormap parameter is called 'cmap'",
    'q24': "subplots(rows, cols) - so subplots(2, 3)",
    'q25': "The figure size parameter is 'figsize'",
}


def get_question_number(func_name):
    """Extract question number from function name."""
    match = re.match(r'q(\d+)', func_name)
    return int(match.group(1)) if match else 0


def run_question(func):
    """Run a single question and return (success, error_message)."""
    try:
        func()
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


def main():
    show_hint = '--hint' in sys.argv
    show_all = '--all' in sys.argv

    # Import questions
    try:
        import importlib.util
        import os
        path = os.path.join(os.path.dirname(__file__), '01_python_review.py')
        spec = importlib.util.spec_from_file_location("review", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ALL_QUESTIONS = module.ALL_QUESTIONS
    except Exception as e:
        print(f"Error loading questions: {e}")
        print("Make sure you're running from the quizzes directory.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  INTRO TO PYTHON - REVIEW")
    print("=" * 60 + "\n")

    passed = 0
    first_failure = None

    for q in ALL_QUESTIONS:
        success, error = run_question(q)
        name = q.__name__
        num = get_question_number(name)

        if success:
            passed += 1
            if show_all:
                print(f"  [PASS] {name}")
        else:
            if first_failure is None:
                first_failure = (name, q.__doc__, error, num)
            if show_all:
                print(f"  [FAIL] {name}")

    print()
    print("=" * 60)

    # Progress bar
    total = len(ALL_QUESTIONS)
    bar_width = 40
    filled = int(bar_width * passed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"  Progress: [{bar}] {passed}/{total}")
    print("=" * 60)

    if first_failure:
        name, doc, error, num = first_failure
        print(f"\n  Next up: {name}")
        print(f"  {doc}")
        print(f"\n  Error: {error}")

        if show_hint:
            hint_key = f"q{num:02d}"
            hint = HINTS.get(hint_key, "No hint available.")
            print(f"\n  HINT: {hint}")
        else:
            print(f"\n  Stuck? Run: python run_review.py --hint")

        print(f"\n  Edit 01_python_review.py and fill in the blanks.")
        print()
    else:
        print("""
  You got them all!

  Topics covered:
    - Python lists, dicts, and control flow
    - NumPy array creation and operations
    - Matplotlib basics

  Ready for the next module.
        """)


if __name__ == '__main__':
    main()
