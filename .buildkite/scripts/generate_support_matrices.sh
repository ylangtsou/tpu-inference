#!/bin/bash
set -euo pipefail

ANY_FAILED=false

MODEL_LIST_KEY="model-list"
FEATURE_LIST_KEY="feature-list"

# Note: This script assumes the metadata keys contain newline-separated lists.
# The `mapfile` command reads these lists into arrays, correctly handling spaces.
mapfile -t model_list < <(buildkite-agent meta-data get "${MODEL_LIST_KEY}" --default "")
mapfile -t feature_list < <(buildkite-agent meta-data get "${FEATURE_LIST_KEY}" --default "")
MODEL_STAGES=("UnitTest" "IntegrationTest" "Benchmark")
FEATURE_STAGES=("CorrectnessTest" "PerformanceTest")
MODEL_CATEGORY=("text-only" "multimodel")

# Output CSV files
model_support_matrix_csv="model_support_matrix.csv"
echo "Model,UnitTest,IntegrationTest,Benchmark" > "$model_support_matrix_csv"

feature_support_matrix_csv="feature_support_matrix.csv"
echo "Feature,CorrectnessTest,PerformanceTest" > "$feature_support_matrix_csv"

process_models_by_category() {
    local category="$1"
    local csv_filename="$2" # Pass filename in

    echo "Model,UnitTest,IntegrationTest,Benchmark" > "$csv_filename"

    # Loop through all models for this specific category
    for model in "${model_list[@]}"; do
        row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            # --- NEW KEY FORMAT ---
            # Get result using the new model:category:stage format
            result=$(buildkite-agent meta-data get "${model}:${category}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$csv_filename"
    done
}

process_models() {
    for model in "$@"; do
        row="\"$model\""
        for stage in "${MODEL_STAGES[@]}"; do
            result=$(buildkite-agent meta-data get "${model}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$model_support_matrix_csv"
    done
}

process_features() {
    for feature in "$@"; do
        row="\"$feature\""
        for stage in "${FEATURE_STAGES[@]}"; do
            result=$(buildkite-agent meta-data get "${feature}:${stage}" --default "N/A")
            row="$row,$result"
            if [ "${result}" != "✅" ] && [ "${result}" != "N/A" ] ; then
                ANY_FAILED=true
            fi
        done
        echo "$row" >> "$feature_support_matrix_csv"
    done
}

# Loop through each category and generate its specific CSV
if [ ${#model_list[@]} -gt 0 ]; then
    for category in "${MODEL_CATEGORY[@]}"; do
        category_filename="$category"
        csv_filename="${category_filename}_model_support_matrix.csv"
        
        # Add to our list for later upload and cleanup
        model_csv_files+=("$csv_filename")

        echo "--- Generating matrix for category: $category ---"
        # Generate the CSV file for this category
        process_models_by_category "$category" "$csv_filename"
    done
fi

if [ ${#model_list[@]} -gt 0 ]; then
    process_models "${model_list[@]}"
fi

if [ ${#feature_list[@]} -gt 0 ]; then
    process_features "${feature_list[@]}"
fi

buildkite-agent meta-data set "CI_TESTS_FAILED" "${ANY_FAILED}"

echo "--- Model support matrix ---"
cat "$model_support_matrix_csv"

echo "--- Feature support matrix ---"
cat "$feature_support_matrix_csv"

echo "--- Saving support matrices as Buildkite Artifacts ---"
for csv_file in "${model_csv_files[@]}"; do
    cat "$csv_file"
    buildkite-agent artifact upload "$csv_file"
done

#buildkite-agent artifact upload "$model_support_matrix_csv"
buildkite-agent artifact upload "$feature_support_matrix_csv"
echo "Reports uploaded successfully."

# cleanup
rm "$model_support_matrix_csv" "$feature_support_matrix_csv"
