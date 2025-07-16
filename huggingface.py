from huggingface_hub import create_repo, upload_folder

# Only run this ONCE to create the repo
# create_repo("fake-news-bert", private=False)  # or private=True

# Upload your model folder
upload_folder(
    folder_path="models/bert_fake_news_model",     # make sure this path is correct
    repo_id="maheen21/fake-news-bert",         # your Hugging Face username/repo
    repo_type="model"
)
