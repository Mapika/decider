"""Upload the trained model + card to the Hub.  Needs HF_TOKEN in env (write scope).
usage: HF_TOKEN=... .venv312/bin/python scripts_upload_hf.py runs/release/decider-2b <user>/<repo>"""
import os, sys, shutil
from huggingface_hub import HfApi
src, repo = sys.argv[1], sys.argv[2]          # src = a staged release folder, e.g. runs/release/decider-2b
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model")
print("uploaded to https://huggingface.co/" + repo)
