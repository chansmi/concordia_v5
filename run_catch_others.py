import os
import subprocess
from pathlib import Path
import concurrent.futures
import sys

# Define paths
agent_dir = Path('concordia/factory/agent')
compiled_jsons_dir = Path('compiled_jsons')

# Get list of agents and existing JSONs
agents = [f.stem for f in agent_dir.glob('*.py') if f.stem != '__init__']
existing_jsons = [f.stem.split('__')[0] for f in compiled_jsons_dir.glob('*.json')]

# Create list of agents without corresponding JSONs
agents_to_process = [agent for agent in agents if agent not in existing_jsons]

total_agents = len(agents_to_process)
processed_agents = 0

def update_progress():
    progress = int(processed_agents * 100 / total_agents)
    bar = '#' * (progress // 2)
    sys.stdout.write(f"\rProgress: [{bar:<50}] {progress}%")
    sys.stdout.flush()

def process_agent(agent):
    global processed_agents
    print(f"Processing agent: {agent}")

    command = [
        "python",
        "examples/modular/launch_concordia_challenge_evaluation.py",
        f"--agent={agent}",
        "--api_type=together_ai",
        "--model=google/gemma-2-27b-it",
        "--num_repetitions_per_scenario=1"
    ]

    try:
        subprocess.run(command, check=True, env={**os.environ, 'PYTHONSAFEPATH': '1'})
        print(f"Successfully processed {agent}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {agent}: {e}")

    processed_agents += 1
    update_progress()

# Use ThreadPoolExecutor to run tasks in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_agent, agent) for agent in agents_to_process]
    concurrent.futures.wait(futures)

print("\nAll agents processed")
