# Attention Asymmetry in AI Layoff Discourse on X

Code and data for the paper:
"Attention Asymmetry in AI Layoff Discourse on X: 
A Computational Analysis of Capital vs Labour Amplification"

Joy Bose, Independent Researcher, Bengaluru, India


## What this repo contains

- Data collection scripts for X (Twitter) API v2 and Reddit public API
- Statistical analysis pipeline (Mann-Whitney U, bootstrap CI, OLS regression)
- Figure generation code
- Aggregate statistics (raw tweet data cannot be shared per X Terms of Service)

## Requirements

pip install -r requirements.txt

## Replication

### Step 1: Collect data
Add your X API bearer token to .env:
BEARER_TOKEN=your_token_here

python collection/collect.py --live

### Step 2: Collect Reddit data (no credentials needed)
python collection/reddit_collect_nocreds.py

### Step 3: Run analysis
python analysis/analyse.py
python analysis/regression_analysis.py

### Step 4: Generate figures
python analysis/visualise.py

## Data availability

Raw tweet data cannot be shared due to X Terms of Service.
Reddit data (data/reddit_posts_clean.csv) is included.
Aggregate engagement statistics are in data/aggregate_stats.csv.

Tweet IDs and Reddit post data are available at https://huggingface.co/datasets/joyboseroy/ai-layoff-discourse-amplification. Raw tweet text cannot be shared per X Terms of Service; full tweet objects can be retrieved by rehydrating the provided IDs using the X API v2.

## Paper

arXiv: https://arxiv.org/abs/2605.29367

## Citation

@misc{bose2026attention,
  title={Attention Asymmetry in AI Layoff Discourse on X: 
         A Computational Analysis of Capital vs Labour Amplification},
  author={Bose, Joy},
  year={2026},
  eprint={2605.29367},
  archivePrefix={arXiv},
  primaryClass={cs.SI},
  url={https://arxiv.org/abs/2605.29367}
}

## License
MIT
