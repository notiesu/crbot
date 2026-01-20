aws s3 cp "s3://1izj1pww36/cr-checkpts/ppo_rewardv2_long/output" \
    "./outputs/ppo_rewardv2_long/output" \
    --region us-ca-2 \
    --endpoint-url https://s3api-us-ca-2.runpod.io \
    --recursive \
    --only-show-errors || {
    echo "Error downloading outputs for ppo_rewardv2_long"
}