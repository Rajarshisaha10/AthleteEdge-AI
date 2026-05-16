import argparse
import os
import subprocess
import sys


BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_step(name, command):
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=BASE, check=True)


def require_file(path, message):
    if not os.path.exists(path):
        raise FileNotFoundError(message)


def main():
    parser = argparse.ArgumentParser(
        description="Set up AthleteEdge AI by installing dependencies and building model files."
    )
    parser.add_argument("--skip-install", action="store_true", help="Do not install requirements.txt.")
    parser.add_argument("--skip-structured", action="store_true", help="Do not train the structured ML models.")
    parser.add_argument("--skip-text", action="store_true", help="Do not generate/train the text model.")
    args = parser.parse_args()

    requirements_path = os.path.join(BASE, "requirements.txt")
    structured_dataset_path = os.path.join(BASE, "dataset.csv")

    if not args.skip_install:
        require_file(requirements_path, "requirements.txt was not found.")
        run_step("Install dependencies", [PYTHON, "-m", "pip", "install", "-r", requirements_path])

    if not args.skip_structured:
        require_file(
            structured_dataset_path,
            "dataset.csv was not found. Add the structured dataset before running structured training.",
        )
        run_step("Train structured injury models", [PYTHON, "train_model.py"])

    if not args.skip_text:
        run_step("Generate text injury dataset", [PYTHON, "generate_dataset.py"])
        run_step("Train text injury model", [PYTHON, "text_train.py"])

    print("\nSetup complete.")
    print("Run the app with:")
    print(f"  {PYTHON} server.py")


if __name__ == "__main__":
    main()
