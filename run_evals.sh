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

# Get all agent names
agent_files=$(ls concordia/factory/agent/*.py)

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
    PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py --agent="$agent_name" --api_type=together_ai --model=google/gemma-2-27b-it --num_repetitions_per_scenario=2 --output_dir="evaluations/$sanitized_agent_name"

    # Check if all scenarios were successful
    all_scenarios_successful=true
    for scenario in "${scenarios[@]}"; do
        json_file="evaluations/$sanitized_agent_name/${sanitized_agent_name}__google_gemma-2-27b-it__all-mpnet-base-v2__only_${scenario}.json"
        if [ ! -f "$json_file" ]; then
            echo "Error: Missing JSON file for scenario $scenario"
            all_scenarios_successful=false
        else
            # Check if the JSON file contains non-zero scores
            if ! grep -q '"focal_per_capita_score": 0' "$json_file" &&
               ! grep -q '"background_per_capita_score": 0' "$json_file" &&
               ! grep -q '"ungrouped_per_capita_score": 0' "$json_file"; then
                echo "Scenario $scenario completed successfully"
            else
                echo "Error: Zero scores found in $json_file"
                all_scenarios_successful=false
            fi
        fi
    done

    if [ "$all_scenarios_successful" = false ]; then
        echo "Re-running launch_concordia_challenge_evaluation.py for $sanitized_agent_name due to errors"
        PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py --agent="$agent_name" --api_type=together_ai --model=google/gemma-2-27b-it --num_repetitions_per_scenario=2 --output_dir="evaluations/$sanitized_agent_name"
    fi

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
}

# Process agents concurrently
for agent_file in $agent_files; do
    process_agent "$agent_file" &
done

# Wait for all background processes to finish
wait

echo "Running calculate_ratings.py"
PYTHONSAFEPATH=1 python examples/modular/calculate_ratings.py --model=google/gemma-2-27b-it --embedder=all-mpnet-base-v2 --agents $(ls -d evaluations/*/ | sed 's#evaluations/##' | sed 's#/##')

echo "Script completed"
