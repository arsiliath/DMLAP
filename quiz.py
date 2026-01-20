#!/usr/bin/env python3
"""
DMLAP Course Review Quiz
========================
Test your knowledge of Data and Machine Learning for Artistic Practice.

Run with: python quiz.py
"""

import random

QUESTIONS = [
    # === PYTHON FUNDAMENTALS ===
    {
        "category": "Python Fundamentals",
        "question": "In Python, how do you define code blocks (like the body of an if statement)?",
        "options": [
            "A) Using curly braces {}",
            "B) Using indentation",
            "C) Using parentheses ()",
            "D) Using the 'end' keyword"
        ],
        "answer": "B",
        "explanation": "Python uses indentation (typically 4 spaces) to define code blocks, unlike JavaScript which uses curly braces."
    },
    {
        "category": "Python Fundamentals",
        "question": "What does my_list[-1] return?",
        "options": [
            "A) An error",
            "B) The first element",
            "C) The last element",
            "D) A reversed list"
        ],
        "answer": "C",
        "explanation": "Negative indexing in Python accesses elements from the end. -1 is the last element, -2 is second to last, etc."
    },
    {
        "category": "Python Fundamentals",
        "question": "Which data structure uses key-value pairs?",
        "options": [
            "A) List",
            "B) Tuple",
            "C) Set",
            "D) Dictionary"
        ],
        "answer": "D",
        "explanation": "Dictionaries store data as key-value pairs, accessed like my_dict['key']."
    },
    {
        "category": "Python Fundamentals",
        "question": "What is the difference between a list and a tuple?",
        "options": [
            "A) Tuples can only contain numbers",
            "B) Lists are immutable, tuples are mutable",
            "C) Tuples are immutable, lists are mutable",
            "D) There is no difference"
        ],
        "answer": "C",
        "explanation": "Tuples are immutable (cannot be changed after creation), while lists are mutable."
    },
    {
        "category": "Python Fundamentals",
        "question": "What does [x**2 for x in range(5)] produce?",
        "options": [
            "A) [0, 1, 2, 3, 4]",
            "B) [1, 4, 9, 16, 25]",
            "C) [0, 1, 4, 9, 16]",
            "D) [0, 2, 4, 6, 8]"
        ],
        "answer": "C",
        "explanation": "This is a list comprehension that squares each number. range(5) gives 0,1,2,3,4, and squaring gives 0,1,4,9,16."
    },

    # === NUMPY ===
    {
        "category": "NumPy",
        "question": "What does np.zeros((3, 4)) create?",
        "options": [
            "A) A 1D array of 12 zeros",
            "B) A 3x4 matrix filled with zeros",
            "C) A 4x3 matrix filled with zeros",
            "D) An error - you need np.zeros([3, 4])"
        ],
        "answer": "B",
        "explanation": "np.zeros((rows, cols)) creates a matrix of the specified shape filled with zeros."
    },
    {
        "category": "NumPy",
        "question": "What is 'broadcasting' in NumPy?",
        "options": [
            "A) Sending arrays over a network",
            "B) Converting arrays to lists",
            "C) Applying operations to arrays of different shapes",
            "D) Printing arrays to the console"
        ],
        "answer": "C",
        "explanation": "Broadcasting allows NumPy to perform operations on arrays with different shapes by automatically expanding dimensions."
    },
    {
        "category": "NumPy",
        "question": "If arr has shape (100, 28, 28, 3), what does arr.shape represent for images?",
        "options": [
            "A) 100 grayscale images of 28x28 pixels",
            "B) 100 RGB images of 28x28 pixels",
            "C) 3 images of 100x28x28 pixels",
            "D) 28 images with 100 channels"
        ],
        "answer": "B",
        "explanation": "Shape (batch, height, width, channels) - so 100 images, 28 pixels tall, 28 wide, 3 color channels (RGB)."
    },
    {
        "category": "NumPy",
        "question": "What does the @ operator do with matrices?",
        "options": [
            "A) Element-wise multiplication",
            "B) Matrix multiplication",
            "C) Array concatenation",
            "D) Power operation"
        ],
        "answer": "B",
        "explanation": "The @ operator performs matrix multiplication. Use * for element-wise multiplication."
    },
    {
        "category": "NumPy",
        "question": "Why would you set np.random.seed(42)?",
        "options": [
            "A) To make random numbers truly random",
            "B) To make results reproducible",
            "C) To generate 42 random numbers",
            "D) To speed up random number generation"
        ],
        "answer": "B",
        "explanation": "Setting a random seed ensures the same sequence of random numbers is generated each time, making experiments reproducible."
    },

    # === MATPLOTLIB ===
    {
        "category": "Matplotlib",
        "question": "What function displays an image in matplotlib?",
        "options": [
            "A) plt.plot()",
            "B) plt.show()",
            "C) plt.imshow()",
            "D) plt.display()"
        ],
        "answer": "C",
        "explanation": "plt.imshow() displays images. plt.plot() is for line plots, and plt.show() displays the figure."
    },
    {
        "category": "Matplotlib",
        "question": "When displaying a single-channel (grayscale) image, what argument prevents false colors?",
        "options": [
            "A) color='gray'",
            "B) cmap='gray'",
            "C) grayscale=True",
            "D) mode='L'"
        ],
        "answer": "B",
        "explanation": "Use cmap='gray' to display grayscale images properly. Without it, matplotlib applies a colormap."
    },

    # === PYTORCH FUNDAMENTALS ===
    {
        "category": "PyTorch",
        "question": "What is a PyTorch tensor?",
        "options": [
            "A) A type of neural network",
            "B) A multi-dimensional array similar to NumPy arrays",
            "C) A function for training models",
            "D) A visualization tool"
        ],
        "answer": "B",
        "explanation": "Tensors are PyTorch's fundamental data structure, similar to NumPy arrays but with GPU support and automatic differentiation."
    },
    {
        "category": "PyTorch",
        "question": "What does model.to('cuda') do?",
        "options": [
            "A) Saves the model to disk",
            "B) Moves the model to GPU memory",
            "C) Converts the model to CUDA format",
            "D) Enables CUDA debugging"
        ],
        "answer": "B",
        "explanation": "to('cuda') moves the model to NVIDIA GPU memory for faster computation. Use 'mps' for Apple Silicon."
    },
    {
        "category": "PyTorch",
        "question": "What is the purpose of a DataLoader?",
        "options": [
            "A) To download datasets from the internet",
            "B) To batch data and handle shuffling during training",
            "C) To save trained models",
            "D) To visualize training progress"
        ],
        "answer": "B",
        "explanation": "DataLoader handles batching, shuffling, and parallel data loading during training."
    },
    {
        "category": "PyTorch",
        "question": "What does optimizer.zero_grad() do in a training loop?",
        "options": [
            "A) Resets the model weights to zero",
            "B) Clears accumulated gradients from previous iterations",
            "C) Stops gradient computation",
            "D) Initializes the optimizer"
        ],
        "answer": "B",
        "explanation": "Gradients accumulate by default in PyTorch. zero_grad() clears them before each training step."
    },

    # === NEURAL NETWORKS ===
    {
        "category": "Neural Networks",
        "question": "What is MNIST commonly used for?",
        "options": [
            "A) Face recognition",
            "B) Handwritten digit classification",
            "C) Audio processing",
            "D) Text generation"
        ],
        "answer": "B",
        "explanation": "MNIST contains 70,000 grayscale images of handwritten digits (0-9). It's often called the 'Hello World' of machine learning."
    },
    {
        "category": "Neural Networks",
        "question": "What is an epoch in training?",
        "options": [
            "A) A single forward pass",
            "B) One complete pass through the entire training dataset",
            "C) The time it takes to train a model",
            "D) A type of optimizer"
        ],
        "answer": "B",
        "explanation": "One epoch = the model has seen every training example once. Training typically requires multiple epochs."
    },
    {
        "category": "Neural Networks",
        "question": "What does the learning rate control?",
        "options": [
            "A) How fast data is loaded",
            "B) The size of weight updates during training",
            "C) The number of layers in the network",
            "D) The batch size"
        ],
        "answer": "B",
        "explanation": "Learning rate determines how much weights are adjusted based on gradients. Too high = unstable, too low = slow learning."
    },
    {
        "category": "Neural Networks",
        "question": "Why do we split data into training and test sets?",
        "options": [
            "A) To make training faster",
            "B) To evaluate how well the model generalizes to unseen data",
            "C) Because the dataset is too large",
            "D) To reduce memory usage"
        ],
        "answer": "B",
        "explanation": "The test set evaluates generalization - how well the model performs on data it wasn't trained on."
    },

    # === CNNs ===
    {
        "category": "CNNs",
        "question": "What is the main advantage of CNNs for image tasks?",
        "options": [
            "A) They require less data",
            "B) They automatically learn spatial hierarchies and features",
            "C) They train faster than other networks",
            "D) They don't need GPUs"
        ],
        "answer": "B",
        "explanation": "CNNs learn local patterns (edges, textures) in early layers and combine them into complex features in deeper layers."
    },
    {
        "category": "CNNs",
        "question": "What does a pooling layer do?",
        "options": [
            "A) Adds more features",
            "B) Reduces spatial dimensions while keeping important features",
            "C) Connects all neurons together",
            "D) Normalizes the output"
        ],
        "answer": "B",
        "explanation": "Pooling (e.g., MaxPool) reduces the spatial size, reducing computation and helping with translation invariance."
    },

    # === GANs ===
    {
        "category": "GANs",
        "question": "What are the two main components of a GAN?",
        "options": [
            "A) Encoder and Decoder",
            "B) Generator and Discriminator",
            "C) Input and Output layers",
            "D) Training and Testing networks"
        ],
        "answer": "B",
        "explanation": "Generator creates fake images, Discriminator tries to distinguish real from fake. They train adversarially."
    },
    {
        "category": "GANs",
        "question": "What is the 'latent space' in a GAN?",
        "options": [
            "A) The space where training happens",
            "B) A random vector space that gets transformed into images",
            "C) The output image dimensions",
            "D) The discriminator's memory"
        ],
        "answer": "B",
        "explanation": "The latent space (z) is a random vector that the generator transforms into images. Interpolating in this space creates smooth transitions."
    },
    {
        "category": "GANs",
        "question": "What is Pix2Pix used for?",
        "options": [
            "A) Generating random images",
            "B) Image-to-image translation with paired data",
            "C) Text-to-image generation",
            "D) Image classification"
        ],
        "answer": "B",
        "explanation": "Pix2Pix is a conditional GAN for paired image translation (e.g., sketch to photo, edges to image)."
    },
    {
        "category": "GANs",
        "question": "In DCGAN, what does 'DC' stand for?",
        "options": [
            "A) Direct Connection",
            "B) Deep Convolutional",
            "C) Data Compression",
            "D) Dual Channel"
        ],
        "answer": "B",
        "explanation": "DCGAN = Deep Convolutional GAN, which uses convolutional layers for more stable image generation."
    },

    # === AUDIO ===
    {
        "category": "Audio",
        "question": "What is a spectrogram?",
        "options": [
            "A) A type of audio file format",
            "B) A visual representation of frequencies over time",
            "C) A neural network for audio",
            "D) A sound recording device"
        ],
        "answer": "B",
        "explanation": "A spectrogram shows how the frequency content of audio changes over time - x-axis is time, y-axis is frequency."
    },
    {
        "category": "Audio",
        "question": "What type of neural network layer works well for audio waveforms?",
        "options": [
            "A) 2D Convolutional layers",
            "B) 1D Convolutional layers",
            "C) Recurrent layers only",
            "D) Pooling layers only"
        ],
        "answer": "B",
        "explanation": "1D convolutions work directly on the time dimension of audio waveforms, detecting temporal patterns."
    },

    # === LANGUAGE MODELS ===
    {
        "category": "Language Models",
        "question": "What does 'temperature' control in text generation?",
        "options": [
            "A) How fast the model runs",
            "B) The randomness/creativity of the output",
            "C) The length of generated text",
            "D) The grammar accuracy"
        ],
        "answer": "B",
        "explanation": "Higher temperature = more random/creative output. Lower temperature = more deterministic/focused output."
    },
    {
        "category": "Language Models",
        "question": "What is fine-tuning a language model?",
        "options": [
            "A) Training from scratch on new data",
            "B) Adjusting a pre-trained model on domain-specific data",
            "C) Reducing the model size",
            "D) Fixing bugs in the model"
        ],
        "answer": "B",
        "explanation": "Fine-tuning adapts a pre-trained model to specific tasks or domains without training from scratch."
    },
    {
        "category": "Language Models",
        "question": "What is a tokenizer?",
        "options": [
            "A) A tool that generates random tokens for security",
            "B) A component that converts text to numerical tokens",
            "C) A type of neural network layer",
            "D) A debugging tool"
        ],
        "answer": "B",
        "explanation": "Tokenizers convert text into numerical tokens that models can process, and vice versa."
    },

    # === DIFFUSION MODELS ===
    {
        "category": "Diffusion Models",
        "question": "How do diffusion models generate images?",
        "options": [
            "A) By copying existing images",
            "B) By iteratively denoising random noise",
            "C) By combining image patches",
            "D) By training two networks adversarially"
        ],
        "answer": "B",
        "explanation": "Diffusion models start with random noise and gradually denoise it into a coherent image over many steps."
    },
    {
        "category": "Diffusion Models",
        "question": "What is ControlNet used for?",
        "options": [
            "A) Controlling training speed",
            "B) Adding spatial control to image generation (pose, edges, depth)",
            "C) Network security",
            "D) Controlling model size"
        ],
        "answer": "B",
        "explanation": "ControlNet adds conditioning to diffusion models, allowing control via poses, edges, depth maps, etc."
    },
    {
        "category": "Diffusion Models",
        "question": "What is Stable Diffusion?",
        "options": [
            "A) A GAN architecture",
            "B) A text-to-image diffusion model",
            "C) An audio generation model",
            "D) A training optimization technique"
        ],
        "answer": "B",
        "explanation": "Stable Diffusion is a popular open-source text-to-image model that uses diffusion for high-quality image generation."
    },
    {
        "category": "Diffusion Models",
        "question": "What does a 'scheduler' do in diffusion models?",
        "options": [
            "A) Schedules when to train the model",
            "B) Controls the denoising process and number of steps",
            "C) Manages GPU memory",
            "D) Schedules data loading"
        ],
        "answer": "B",
        "explanation": "Schedulers control how noise is added/removed during the diffusion process. Different schedulers offer speed/quality tradeoffs."
    },

    # === HUGGING FACE ===
    {
        "category": "Hugging Face",
        "question": "What is Hugging Face Hub?",
        "options": [
            "A) A code editor",
            "B) A repository for sharing ML models and datasets",
            "C) A GPU cloud service",
            "D) A machine learning framework"
        ],
        "answer": "B",
        "explanation": "Hugging Face Hub hosts thousands of pre-trained models and datasets that can be easily downloaded and used."
    },
    {
        "category": "Hugging Face",
        "question": "What library do you use for diffusion models from Hugging Face?",
        "options": [
            "A) transformers",
            "B) datasets",
            "C) diffusers",
            "D) accelerate"
        ],
        "answer": "C",
        "explanation": "The diffusers library provides easy-to-use pipelines for diffusion models like Stable Diffusion."
    },
]


def run_quiz(num_questions=15):
    """Run an interactive quiz."""
    print("\n" + "=" * 60)
    print("  DMLAP COURSE REVIEW QUIZ")
    print("  Data and Machine Learning for Artistic Practice")
    print("=" * 60)
    print(f"\nThis quiz will test your knowledge with {num_questions} questions.")
    print("Answer by typing the letter (A, B, C, or D) and pressing Enter.")
    print("Type 'q' to quit at any time.\n")

    # Select random questions
    selected = random.sample(QUESTIONS, min(num_questions, len(QUESTIONS)))

    score = 0
    answered = 0

    for i, q in enumerate(selected, 1):
        print("-" * 60)
        print(f"Question {i}/{len(selected)} [{q['category']}]")
        print("-" * 60)
        print(f"\n{q['question']}\n")

        for option in q['options']:
            print(f"  {option}")

        while True:
            answer = input("\nYour answer: ").strip().upper()

            if answer == 'Q':
                print(f"\n{'=' * 60}")
                print(f"Quiz ended early. You scored {score}/{answered} ({100*score//max(answered,1)}%)")
                print("=" * 60)
                return

            if answer in ['A', 'B', 'C', 'D']:
                break
            print("Please enter A, B, C, or D (or Q to quit)")

        answered += 1

        if answer == q['answer']:
            score += 1
            print("\n  Correct!")
        else:
            print(f"\n  Incorrect. The answer was {q['answer']}.")

        print(f"  {q['explanation']}")
        print(f"\n  Current score: {score}/{answered}")

    # Final results
    percentage = 100 * score // len(selected)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Score: {score}/{len(selected)} ({percentage}%)\n")

    if percentage >= 90:
        print("  Excellent! You have a strong grasp of the material.")
    elif percentage >= 70:
        print("  Good job! You know the fundamentals well.")
        print("  Consider reviewing the topics you missed.")
    elif percentage >= 50:
        print("  Not bad, but there's room for improvement.")
        print("  Review the course notebooks for topics you found challenging.")
    else:
        print("  You might want to review the course material more thoroughly.")
        print("  Start with the fundamentals and work your way up.")

    # Show categories that need work
    print("\n" + "-" * 60)
    print("  TOPIC BREAKDOWN")
    print("-" * 60)

    categories = {}
    for i, q in enumerate(selected):
        cat = q['category']
        if cat not in categories:
            categories[cat] = {'correct': 0, 'total': 0}
        categories[cat]['total'] += 1

    # This is a simplified breakdown - in a real quiz you'd track per-question
    print("\n  Categories covered in this quiz:")
    for cat in sorted(categories.keys()):
        print(f"    - {cat}")

    print("\n" + "=" * 60)


def list_topics():
    """List all topics covered in the quiz."""
    print("\n" + "=" * 60)
    print("  TOPICS COVERED IN THIS QUIZ")
    print("=" * 60)

    categories = {}
    for q in QUESTIONS:
        cat = q['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    print(f"\nTotal questions available: {len(QUESTIONS)}\n")

    for cat in sorted(categories.keys()):
        print(f"  {cat}: {categories[cat]} questions")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--topics":
            list_topics()
        elif sys.argv[1] == "--all":
            run_quiz(num_questions=len(QUESTIONS))
        elif sys.argv[1].isdigit():
            run_quiz(num_questions=int(sys.argv[1]))
        else:
            print("Usage: python quiz.py [OPTIONS]")
            print("  (no args)  - Run quiz with 15 random questions")
            print("  --topics   - List all topics covered")
            print("  --all      - Run quiz with all questions")
            print("  <number>   - Run quiz with specified number of questions")
    else:
        run_quiz()
