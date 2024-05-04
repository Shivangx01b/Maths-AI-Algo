import pandas as pd
import random

# Define a wide range of operations including arithmetic, geometry, algebra, calculus, and probability with correct placeholders
comprehensive_operations = [
    # Basic Arithmetic
    ("What is {} plus {}?", lambda x, y: x + y),
    ("What is {} minus {}?", lambda x, y: x - y),
    ("What is {} times {}?", lambda x, y: x * y),
    ("What is {} divided by {}?", lambda x, y: x // y),
    # Geometry
    ("What is the perimeter of a square with side length {}?", lambda x: 4 * x),
    ("What is the area of a rectangle with width {} and height {}?", lambda x, y: x * y),
    ("What is the area of a circle with radius {}?", lambda x: round(3.14 * x * x, 2)),
    # Algebra
    ("Solve for x in the equation x + {} = {}.", lambda x, y: y - x),
    ("What is the solution for x in the equation 2x - {} = {}?", lambda x, y: (y + x) // 2),
    # Calculus
    ("Find the derivative of {}x^2 at x = {}.", lambda x, y: 2 * x * y),
    ("Calculate the integral of {}x from 0 to {}.", lambda x, y: (x * y**2) // 2),
    # Probability
    ("What is the probability of rolling a die and getting a number greater than {}?", lambda x: f"{round((6 - x) / 6 * 100, 2)}%"),
    ("A bag contains {} red and {} blue balls. What is the probability of drawing a red ball?", lambda x, y: f"{round(x / (x + y) * 100, 2)}%"),
    # Statistics
    ("Calculate the average of the numbers {}, {}, and {}.", lambda x, y, z: (x + y + z) // 3),
    ("Find the median of {}, {}, and {}.", lambda x, y, z: sorted([x, y, z])[1])
]

# Generate questions and answers
full_questions = []
full_answers = []

for operation in comprehensive_operations:
    op_text, func = operation
    for _ in range(100):  # Generating 10,000 questions for each operation type
        nums = [random.randint(1, 100) for _ in range(op_text.count("{}"))]  # Generate random numbers for each placeholder
        
        question = op_text.format(*nums)  # Format question with numbers
        answer = str(func(*nums))  # Calculate answer using the operation function
        
        full_questions.append(question)
        full_answers.append(answer)

# Creating DataFrame from the questions and answers
full_dataset = pd.DataFrame({
    "Question": full_questions,
    "Answer": full_answers
})

# Shuffle the DataFrame
full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)

# Save the dataset to a CSV file
full_dataset_path = "Comprehensive_Maths_Dataset_Shuffle.csv"
full_dataset.to_csv(full_dataset_path, index=False)