#!/bin/bash

# Function to sanitize agent names
sanitize_agent_name() {
    echo "$1" | sed 's/[^a-zA-Z0-9_]/_/g'
}

# List of scenarios
scenarios=(
    "labor_collective_action__fixed_rule_boss_0"
    "labor_collective_action__rational_boss_0"
    "labor_collective_action__paranoid_boss_0"
    "pub_coordination_0"
    "pub_coordination_friendships_0"
    "haggling_0"
    "haggling_1"
    "haggling_multi_item_0"
    "reality_show_circa_2003_prisoners_dilemma_0"
    "reality_show_circa_2003_stag_hunt_0"
)

# Create evaluations directory
mkdir -p evaluations

# Function to process a single agent
process_agent() {
    agent_file=$1
    agent_name=$(basename "$agent_file" .py)

    # Skip __init__ and test files
    if [[ "$agent_name" == "__init__" || "$agent_name" == *"test"* ]]; then
        return
    fi

    sanitized_agent_name=$(sanitize_agent_name "$agent_name")

    echo "Processing agent: $agent_name (sanitized: $sanitized_agent_name)"

    # Create folder for agent if it doesn't exist
    mkdir -p "evaluations/$sanitized_agent_name"
    echo "Created folder: evaluations/$sanitized_agent_name"

    echo "Running launch_concordia_challenge_evaluation.py for $sanitized_agent_name"
    PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py --agent="$agent_name" --api_type=together_ai --model=google/gemma-2-27b-it --num_repetitions_per_scenario=1

    # Move the output files to the correct directory
    for scenario in "${scenarios[@]}"; do
        mv "${sanitized_agent_name}__google_gemma-2-27b-it__all-mpnet-base-v2__only_${scenario}.json" "evaluations/$sanitized_agent_name/" 2>/dev/null
    done

    echo "Combining JSON files for agent: $sanitized_agent_name"
    # Combine all JSON files for the agent
    combined_json_file="evaluations/${sanitized_agent_name}/${sanitized_agent_name}__google_gemma-2-27b-it__all-mpnet-base-v2.json"
    echo "[" > "$combined_json_file"
    first_file=true
    for json_file in evaluations/$sanitized_agent_name/*__only_*.json; do
        if [ "$first_file" = true ]; then
            first_file=false
        else
            echo "," >> "$combined_json_file"
        fi
        cat "$json_file" >> "$combined_json_file"
    done
    echo "]" >> "$combined_json_file"
    echo "Combined JSON file created: $combined_json_file"

    # Signal that this agent has been processed
    echo "DONE" > "/tmp/agent_${sanitized_agent_name}_done"
}

# Get all agent files
agent_files=$(ls concordia/factory/agent/*.py)
total_agents=$(echo "$agent_files" | wc -l)
processed_agents=0

# Function to update progress
update_progress() {
    processed_agents=$(ls /tmp/agent_*_done 2>/dev/null | wc -l)
    progress=$((processed_agents * 100 / total_agents))
    printf "\rProgress: [%-50s] %d%%" $(printf '#%.0s' $(seq 1 $((progress / 2)))) $progress
}

# Process agents in parallel, 10 at a time
for agent_file in $agent_files; do
    # Run the process_agent function in the background
    (process_agent "$agent_file") &
    
    # Limit to 10 parallel processes
    while [ $(jobs -r -p | wc -l) -ge 10 ]; do
        sleep 1
        update_progress
    done
done

# Wait for all background jobs to finish and update progress
while [ $processed_agents -lt $total_agents ]; do
    sleep 1
    update_progress
done

echo -e "\nAll agents processed."

echo "Running calculate_ratings.py"
PYTHONSAFEPATH=1 python examples/modular/calculate_ratings.py --model=google/gemma-2-27b-it --embedder=all-mpnet-base-v2 --agents $(ls -d evaluations/*/ | sed 's#evaluations/##' | sed 's#/##')

echo "Script completed"

# Clean up temporary files
rm /tmp/agent_*_done
