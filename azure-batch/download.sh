#!/bin/bash

# Install dependencies
dpkg -s curl >/dev/null 2>&1
if [ ! $? -eq 0 ]; then
    echo "Install curl."
    sudo apt-get update && sudo apt-get install -y curl
fi
dpkg -s jq >/dev/null 2>&1
if [ ! $? -eq 0 ]; then
    echo "Install jq."
    sudo apt-get update && sudo apt-get install -y jq
fi

# Check if PERSONAL_ACCESS_TOKEN is set
if [ -z "$PERSONAL_ACCESS_TOKEN" ]; then
    echo "***********************************************"
    echo "Error: The environment variable PERSONAL_ACCESS_TOKEN is not set."
    echo "Please set it before running this script."
    echo "You can set it by running the following command:"
    echo "export PERSONAL_ACCESS_TOKEN='your_token_here'"
    echo "***********************************************"
    exit 1
else
    GITHUB_TOKEN=$PERSONAL_ACCESS_TOKEN
fi


# repository information
REPO="PanasonicConnect/VideoMultiAgents"
TAG="for_file_share"
FILES=("egoschema_summary_cache.json" "egoschema_lavila_captions.json" "egoschema_videotree_result.json" "momaqa_summary_cache.json" "nextqa_llava1.5_captions.json" "graph_captions_1500.json")

echo "Fetching release assets..."
ASSETS_JSON=$(curl -sL -H "Authorization: token $GITHUB_TOKEN" \
    "https://api.github.com/repos/$REPO/releases/tags/$TAG")

# JSON check
if [ -z "$ASSETS_JSON" ] || [[ "$ASSETS_JSON" == "null" ]]; then
    echo "Error: Failed to fetch release assets."
    exit 1
fi

# Download files
for FILE in "${FILES[@]}"; do
    if [ ! -e "$FILE" ]; then
        echo "Downloading $FILE..."

        # Get asset ID
        ASSET_ID=$(echo "$ASSETS_JSON" | jq -r --arg FILE "$FILE" '.assets[] | select(.name == $FILE) | .id')

        if [ -n "$ASSET_ID" ] && [ "$ASSET_ID" != "null" ]; then
            # Download asset
            curl -LJO -H "Authorization: token $GITHUB_TOKEN" -H 'Accept: application/octet-stream' \
                "https://api.github.com/repos/$REPO/releases/assets/$ASSET_ID"
        else
            echo "Warning: $FILE not found in the release assets."
        fi
    else
        echo "$FILE already exists."
    fi
done

echo "Download completed."
