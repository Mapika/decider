"""Upload the trained model + card to the Hub.  Needs a write token: HF_TOKEN in the environment, or a saved `hf auth login`.
usage: HF_TOKEN=... .venv312/bin/python scripts_upload_hf.py runs/release/decider-2b <user>/<repo>"""
import os, sys, shutil
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]          # src = a staged release folder, e.g. runs/release/decider-2b
api = HfApi(token=os.environ.get("HF_TOKEN"))          # None: the token saved by `hf auth login`
api.create_repo(repo, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", ignore_patterns=["**/__pycache__/**", "*.pyc"],
                  delete_patterns=["decider/__pycache__/**", "decider/*.py", "README.md", "eval_results.json"])  # replace helper/card, drop pycache
print("uploaded to https://huggingface.co/" + repo)
