import json
from pathlib import Path
import subprocess

def combine_json_files(evaluation_folder, output_folder):
    for agent_folder in evaluation_folder.iterdir():
        if agent_folder.is_dir():
            combined_data = []
            for json_file in agent_folder.glob('*.json'):
                with json_file.open() as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend([item for item in data if item.get('scenario') != 'pub_coordination_mini'])
                    elif isinstance(data, dict) and data.get('scenario') != 'pub_coordination_mini':
                        combined_data.append(data)

            if combined_data:
                focal_agent = combined_data[0]['focal_agent']
                output_file = output_folder / f"{focal_agent}.json"
                with output_file.open('w') as f:
                    json.dump(combined_data, f, indent=2)

def main():
    evaluations_folder = Path('evaluations')
    compiled_folder = Path('compiled_jsons')
    compiled_folder.mkdir(exist_ok=True)

    combine_json_files(evaluations_folder, compiled_folder)

    # Run calculate_ratings.py on the compiled folder
    agent_files = [f.stem for f in compiled_folder.glob('*.json')]
    if agent_files:
        try:
            subprocess.run(['python3', 'examples/modular/calculate_ratings.py',
                            '--agents'] + agent_files, check=True)
        except FileNotFoundError:
            try:
                subprocess.run(['python', 'examples/modular/calculate_ratings.py',
                                '--agents'] + agent_files, check=True)
            except FileNotFoundError:
                print("Error: Neither 'python3' nor 'python' command was found. Please ensure Python is installed and in your system PATH.")
    else:
        print("No agent files found in the compiled_jsons folder.")

if __name__ == "__main__":
    main()
