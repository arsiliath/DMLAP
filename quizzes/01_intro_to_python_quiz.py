#!/usr/bin/env python3
"""
Quiz: Introduction to Python
============================
Fill-in-the-code review quiz for 01-intro-to-python module.
Inspired by Ruby Koans - type the correct code to pass each challenge.

Run with: python 01_intro_to_python_quiz.py
"""

import random

CHALLENGES = [
    # === PYTHON BASICS ===
    {
        "category": "Python Basics",
        "description": "Access the last element of a list using negative indexing",
        "setup": "my_list = [10, 20, 30, 40, 50]",
        "prompt": "last_element = my_list[___]",
        "answer": ["-1"],
        "hint": "Negative indices count from the end",
        "test": lambda ans: ans.strip() == "-1"
    },
    {
        "category": "Python Basics",
        "description": "Get elements at index 1, 2, and 3 using slicing",
        "setup": "nums = [0, 1, 2, 3, 4, 5]",
        "prompt": "middle = nums[___]",
        "answer": ["1:4"],
        "hint": "Slice syntax is [start:end] where end is exclusive",
        "test": lambda ans: ans.strip() == "1:4"
    },
    {
        "category": "Python Basics",
        "description": "Create a list of squares [0, 1, 4, 9, 16] using list comprehension",
        "setup": "",
        "prompt": "squares = [___ for x in range(5)]",
        "answer": ["x**2", "x*x"],
        "hint": "Square x using ** or *",
        "test": lambda ans: ans.strip() in ["x**2", "x*x", "x ** 2", "x * x"]
    },
    {
        "category": "Python Basics",
        "description": "Add an element to the end of a list",
        "setup": "fruits = ['apple', 'banana']",
        "prompt": "fruits.___(\"cherry\")",
        "answer": ["append"],
        "hint": "The method to add to the end of a list",
        "test": lambda ans: ans.strip() == "append"
    },
    {
        "category": "Python Basics",
        "description": "Create a dictionary with 'name' as key and 'Alice' as value",
        "setup": "",
        "prompt": "person = {___: 'Alice'}",
        "answer": ["'name'", "\"name\""],
        "hint": "Dictionary keys can be strings",
        "test": lambda ans: ans.strip() in ["'name'", '"name"']
    },
    {
        "category": "Python Basics",
        "description": "Access the value for key 'age' in a dictionary",
        "setup": "person = {'name': 'Bob', 'age': 25}",
        "prompt": "age = person[___]",
        "answer": ["'age'", "\"age\""],
        "hint": "Use the key as a string",
        "test": lambda ans: ans.strip() in ["'age'", '"age"']
    },
    {
        "category": "Python Basics",
        "description": "Generate numbers 0 through 4",
        "setup": "",
        "prompt": "for i in range(___):",
        "answer": ["5"],
        "hint": "range(n) goes from 0 to n-1",
        "test": lambda ans: ans.strip() == "5"
    },
    {
        "category": "Python Basics",
        "description": "Import numpy with the alias 'np'",
        "setup": "",
        "prompt": "import numpy ___ np",
        "answer": ["as"],
        "hint": "The keyword for creating an alias",
        "test": lambda ans: ans.strip() == "as"
    },
    {
        "category": "Python Basics",
        "description": "Unpack a tuple into two variables",
        "setup": "point = (3, 7)",
        "prompt": "x, ___ = point",
        "answer": ["y"],
        "hint": "Name for the second variable",
        "test": lambda ans: ans.strip().isidentifier() and ans.strip() != "x"
    },
    {
        "category": "Python Basics",
        "description": "Check if a value is in a list",
        "setup": "colors = ['red', 'green', 'blue']",
        "prompt": "has_red = 'red' ___ colors",
        "answer": ["in"],
        "hint": "The membership operator",
        "test": lambda ans: ans.strip() == "in"
    },

    # === NUMPY ===
    {
        "category": "NumPy",
        "description": "Create a 3x3 matrix of zeros",
        "setup": "import numpy as np",
        "prompt": "zeros = np.zeros((___,___))",
        "answer": ["3, 3", "3,3"],
        "hint": "Shape is (rows, cols)",
        "test": lambda ans: ans.replace(" ", "") == "3,3"
    },
    {
        "category": "NumPy",
        "description": "Create an array from 0 to 9",
        "setup": "import numpy as np",
        "prompt": "arr = np.arange(___)",
        "answer": ["10"],
        "hint": "arange(n) gives 0 to n-1",
        "test": lambda ans: ans.strip() == "10"
    },
    {
        "category": "NumPy",
        "description": "Create 5 evenly spaced values from 0 to 1 (inclusive)",
        "setup": "import numpy as np",
        "prompt": "arr = np.linspace(0, 1, ___)",
        "answer": ["5"],
        "hint": "The third argument is the number of values",
        "test": lambda ans: ans.strip() == "5"
    },
    {
        "category": "NumPy",
        "description": "Get the shape of an array",
        "setup": "import numpy as np\narr = np.zeros((4, 5))",
        "prompt": "dimensions = arr.___",
        "answer": ["shape"],
        "hint": "The attribute that shows array dimensions",
        "test": lambda ans: ans.strip() == "shape"
    },
    {
        "category": "NumPy",
        "description": "Perform matrix multiplication",
        "setup": "import numpy as np\na = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6], [7, 8]])",
        "prompt": "result = a ___ b",
        "answer": ["@"],
        "hint": "The operator for matrix multiplication (not *)",
        "test": lambda ans: ans.strip() == "@"
    },
    {
        "category": "NumPy",
        "description": "Transpose a matrix (swap rows and columns)",
        "setup": "import numpy as np\nmatrix = np.array([[1, 2, 3], [4, 5, 6]])",
        "prompt": "transposed = matrix.___",
        "answer": ["T"],
        "hint": "A single letter attribute",
        "test": lambda ans: ans.strip() == "T"
    },
    {
        "category": "NumPy",
        "description": "Set random seed for reproducibility",
        "setup": "import numpy as np",
        "prompt": "np.random.___(42)",
        "answer": ["seed"],
        "hint": "The function to set the random state",
        "test": lambda ans: ans.strip() == "seed"
    },
    {
        "category": "NumPy",
        "description": "Flip an image array vertically (reverse row order)",
        "setup": "import numpy as np\nimg = np.array([[1, 2], [3, 4], [5, 6]])",
        "prompt": "flipped = img[___]",
        "answer": ["::-1"],
        "hint": "Slice that reverses the first axis",
        "test": lambda ans: ans.strip() == "::-1"
    },
    {
        "category": "NumPy",
        "description": "Stack arrays vertically",
        "setup": "import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])",
        "prompt": "stacked = np.___(( a, b))",
        "answer": ["vstack"],
        "hint": "v for vertical",
        "test": lambda ans: ans.strip() == "vstack"
    },
    {
        "category": "NumPy",
        "description": "Get the index of the maximum value",
        "setup": "import numpy as np\narr = np.array([3, 7, 2, 9, 4])",
        "prompt": "max_index = arr.___()",
        "answer": ["argmax"],
        "hint": "arg + max",
        "test": lambda ans: ans.strip() == "argmax"
    },

    # === MATPLOTLIB ===
    {
        "category": "Matplotlib",
        "description": "Display an image array",
        "setup": "import matplotlib.pyplot as plt\nimport numpy as np\nimg = np.random.rand(100, 100)",
        "prompt": "plt.___(img)",
        "answer": ["imshow"],
        "hint": "im + show",
        "test": lambda ans: ans.strip() == "imshow"
    },
    {
        "category": "Matplotlib",
        "description": "Display a grayscale image without false colors",
        "setup": "import matplotlib.pyplot as plt\nimport numpy as np\ngray_img = np.random.rand(100, 100)",
        "prompt": "plt.imshow(gray_img, ___='gray')",
        "answer": ["cmap"],
        "hint": "Short for colormap",
        "test": lambda ans: ans.strip() == "cmap"
    },
    {
        "category": "Matplotlib",
        "description": "Create a figure with 2 rows and 3 columns of subplots",
        "setup": "import matplotlib.pyplot as plt",
        "prompt": "fig, axes = plt.subplots(___, ___)",
        "answer": ["2, 3", "2,3"],
        "hint": "(rows, cols)",
        "test": lambda ans: ans.replace(" ", "") == "2,3"
    },
    {
        "category": "Matplotlib",
        "description": "Set the figure size to 10 inches wide and 5 inches tall",
        "setup": "import matplotlib.pyplot as plt",
        "prompt": "plt.figure(___=(10, 5))",
        "answer": ["figsize"],
        "hint": "fig + size",
        "test": lambda ans: ans.strip() == "figsize"
    },
    {
        "category": "Matplotlib",
        "description": "Create a line plot of x vs y",
        "setup": "import matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(0, 10, 100)\ny = np.sin(x)",
        "prompt": "plt.___(x, y)",
        "answer": ["plot"],
        "hint": "The basic plotting function",
        "test": lambda ans: ans.strip() == "plot"
    },
]


def run_quiz(num_challenges=10):
    """Run the interactive coding quiz."""
    print("\n" + "=" * 60)
    print("  INTRO TO PYTHON - CODE QUIZ")
    print("  Fill in the blanks with the correct Python code")
    print("=" * 60)
    print(f"\nYou'll complete {num_challenges} coding challenges.")
    print("Type your answer and press Enter.")
    print("Type 'hint' for a hint, 'skip' to skip, or 'quit' to exit.\n")

    selected = random.sample(CHALLENGES, min(num_challenges, len(CHALLENGES)))

    score = 0
    skipped = 0
    results_by_category = {}

    for i, c in enumerate(selected, 1):
        cat = c['category']
        if cat not in results_by_category:
            results_by_category[cat] = {'correct': 0, 'total': 0}
        results_by_category[cat]['total'] += 1

        print("-" * 60)
        print(f"Challenge {i}/{len(selected)} [{cat}]")
        print("-" * 60)
        print(f"\n{c['description']}\n")

        if c['setup']:
            print("  # Given:")
            for line in c['setup'].split('\n'):
                print(f"  {line}")
            print()

        print(f"  # Complete this line:")
        print(f"  {c['prompt']}")
        print()

        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            answer = input("  >>> ").strip()

            if answer.lower() == 'quit':
                print(f"\n{'=' * 60}")
                print(f"Quiz ended. Score: {score}/{i-1+skipped} ({skipped} skipped)")
                print("=" * 60)
                return

            if answer.lower() == 'skip':
                skipped += 1
                print(f"\n  Skipped. Answer was: {c['answer'][0]}")
                break

            if answer.lower() == 'hint':
                print(f"\n  Hint: {c['hint']}\n")
                continue

            attempts += 1

            if c['test'](answer):
                score += 1
                results_by_category[cat]['correct'] += 1
                print("\n  Correct!\n")
                break
            else:
                remaining = max_attempts - attempts
                if remaining > 0:
                    print(f"\n  Not quite. {remaining} attempt(s) left.\n")
                else:
                    print(f"\n  The answer was: {c['answer'][0]}\n")

        print(f"  Score: {score}/{i} ({skipped} skipped)")

    # Final results
    answered = len(selected) - skipped
    percentage = 100 * score // answered if answered > 0 else 0

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Correct: {score}/{len(selected)}")
    print(f"  Skipped: {skipped}")
    print(f"  Score: {percentage}%\n")

    if percentage >= 90:
        print("  Excellent! You've mastered the basics.")
    elif percentage >= 70:
        print("  Good work! A few areas to review.")
    elif percentage >= 50:
        print("  Review the intro notebooks for topics you missed.")
    else:
        print("  Recommended: Work through the intro notebooks again.")

    print("\n" + "-" * 60)
    print("  BREAKDOWN BY TOPIC")
    print("-" * 60)

    for cat in ['Python Basics', 'NumPy', 'Matplotlib']:
        if cat in results_by_category:
            r = results_by_category[cat]
            if r['total'] > 0:
                pct = 100 * r['correct'] // r['total']
                status = "OK" if pct >= 70 else "REVIEW"
                print(f"  {cat}: {r['correct']}/{r['total']} [{status}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_quiz(num_challenges=len(CHALLENGES))
        elif sys.argv[1].isdigit():
            run_quiz(num_challenges=int(sys.argv[1]))
        else:
            print("Usage: python 01_intro_to_python_quiz.py [OPTIONS]")
            print("  (no args)  - Run 10 random challenges")
            print("  --all      - Run all 25 challenges")
            print("  <number>   - Run specified number of challenges")
    else:
        run_quiz()
