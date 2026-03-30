"""
Disease Prediction CLI
Run from the project root:  python app.py
"""

import os
import sys

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.predict import predict, get_symptom_list

BANNER = """
╔══════════════════════════════════════╗
║      Disease Prediction System       ║
╚══════════════════════════════════════╝
Enter up to 5 symptoms and get a predicted disease.
Type 'list' to see all valid symptoms, or 'quit' to exit.
"""


def run():
    print(BANNER)

    while True:
        symptoms = []
        print("─" * 40)
        for i in range(1, 6):
            raw = input(f"  Symptom {i} (or press Enter to skip): ").strip()
            if raw.lower() == "quit":
                print("Goodbye.")
                return
            if raw.lower() == "list":
                cols = get_symptom_list()
                print("\nValid symptoms:")
                for j, s in enumerate(cols, 1):
                    print(f"  {j:>3}. {s}")
                print()
                # Restart this round
                symptoms = []
                break
            if raw:
                symptoms.append(raw)

        if not symptoms:
            continue

        result = predict(symptoms)
        print(f"\n  Predicted disease: {result}\n")


if __name__ == "__main__":
    run()
