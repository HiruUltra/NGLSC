"""
Quiz generation service.
Owns the question bank and the sampling logic.
"""
import random
from typing import List

from app.schemas.quiz import QuizQuestion


# ── Question Bank ─────────────────────────────────────────────────────────────

QUESTION_BANK: dict = {
    "Mathematics": [
        {"question": "What is 15 × 12?",
         "options": ["150", "180", "175", "165"], "correct_answer": 1},
        {"question": "Solve: 2x + 5 = 15. What is x?",
         "options": ["5", "7", "10", "3"], "correct_answer": 0},
        {"question": "What is the area of a circle with radius 7 cm? (π = 22/7)",
         "options": ["154 cm²", "144 cm²", "164 cm²", "134 cm²"], "correct_answer": 0},
        {"question": "What is the value of √144?",
         "options": ["10", "12", "14", "16"], "correct_answer": 1},
        {"question": "A triangle with angles 60°, 60°, 60° is:",
         "options": ["Scalene", "Isosceles", "Equilateral", "Right-angled"], "correct_answer": 2},
        {"question": "What is 25% of 200?",
         "options": ["25", "50", "75", "100"], "correct_answer": 1},
        {"question": "Solve: 3² + 4² = ?",
         "options": ["25", "49", "36", "16"], "correct_answer": 0},
        {"question": "What is the next prime number after 7?",
         "options": ["9", "10", "11", "13"], "correct_answer": 2},
    ],
    "Science": [
        {"question": "What is the chemical symbol for water?",
         "options": ["O₂", "H₂O", "CO₂", "H₂"], "correct_answer": 1},
        {"question": "What is the speed of light in vacuum?",
         "options": ["3 × 10⁸ m/s", "3 × 10⁶ m/s", "3 × 10⁷ m/s", "3 × 10⁹ m/s"], "correct_answer": 0},
        {"question": "Which planet is known as the Red Planet?",
         "options": ["Venus", "Jupiter", "Mars", "Saturn"], "correct_answer": 2},
        {"question": "What is the powerhouse of the cell?",
         "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi body"], "correct_answer": 1},
        {"question": "What gas do plants absorb from the atmosphere?",
         "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "correct_answer": 2},
        {"question": "What is the atomic number of Carbon?",
         "options": ["4", "6", "8", "12"], "correct_answer": 1},
        {"question": "Which force keeps us on the ground?",
         "options": ["Magnetic", "Gravity", "Friction", "Tension"], "correct_answer": 1},
        {"question": "What is the boiling point of water at sea level?",
         "options": ["90°C", "100°C", "110°C", "120°C"], "correct_answer": 1},
    ],
    "History": [
        {"question": "In which year did World War II end?",
         "options": ["1943", "1944", "1945", "1946"], "correct_answer": 2},
        {"question": "Who was the first President of the United States?",
         "options": ["Thomas Jefferson", "George Washington", "John Adams", "Benjamin Franklin"], "correct_answer": 1},
        {"question": "The Great Wall of China was built to protect against invasions from:",
         "options": ["Mongols", "Japanese", "Russians", "Indians"], "correct_answer": 0},
        {"question": "Who painted the Mona Lisa?",
         "options": ["Michelangelo", "Raphael", "Leonardo da Vinci", "Donatello"], "correct_answer": 2},
        {"question": "The French Revolution began in which year?",
         "options": ["1776", "1789", "1799", "1804"], "correct_answer": 1},
        {"question": "Which ancient civilization built the pyramids?",
         "options": ["Romans", "Greeks", "Egyptians", "Mayans"], "correct_answer": 2},
        {"question": "Who discovered America in 1492?",
         "options": ["Vasco da Gama", "Christopher Columbus", "Ferdinand Magellan", "Marco Polo"], "correct_answer": 1},
        {"question": "The Renaissance period began in which country?",
         "options": ["France", "Spain", "Italy", "England"], "correct_answer": 2},
    ],
    "General Knowledge": [
        {"question": "What is the capital of France?",
         "options": ["London", "Berlin", "Paris", "Madrid"], "correct_answer": 2},
        {"question": "How many continents are there?",
         "options": ["5", "6", "7", "8"], "correct_answer": 2},
        {"question": "What is the largest ocean on Earth?",
         "options": ["Atlantic", "Indian", "Arctic", "Pacific"], "correct_answer": 3},
        {"question": "Which is the tallest mountain in the world?",
         "options": ["K2", "Mount Everest", "Kangchenjunga", "Lhotse"], "correct_answer": 1},
        {"question": "What is the smallest country in the world?",
         "options": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"], "correct_answer": 1},
        {"question": "Which language is most spoken worldwide?",
         "options": ["Spanish", "English", "Mandarin Chinese", "Hindi"], "correct_answer": 2},
        {"question": "What is the largest desert in the world?",
         "options": ["Sahara", "Arabian", "Gobi", "Antarctica"], "correct_answer": 3},
        {"question": "How many colors are in a rainbow?",
         "options": ["5", "6", "7", "8"], "correct_answer": 2},
    ],
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_quiz(topic: str, num_questions: int) -> List[QuizQuestion]:
    """
    Return a randomly sampled list of quiz questions for *topic*.

    Falls back to General Knowledge when the topic is not recognised.
    Repeats questions if more are requested than available.
    """
    # Case-insensitive partial match
    matching_topic = next(
        (t for t in QUESTION_BANK if topic.lower() in t.lower() or t.lower() in topic.lower()),
        "General Knowledge",
    )

    pool = QUESTION_BANK[matching_topic].copy()

    # Pad pool if caller requests more questions than exist
    if num_questions > len(pool):
        multiplier = (num_questions // len(pool)) + 1
        pool = pool * multiplier

    selected = random.sample(pool, min(num_questions, len(pool)))

    return [
        QuizQuestion(
            id=idx + 1,
            question=q["question"],
            options=q["options"],
            correct_answer=q["correct_answer"],
        )
        for idx, q in enumerate(selected)
    ]
