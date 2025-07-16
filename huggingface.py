from huggingface_hub import upload_folder

upload_folder(
    folder_path="models/bert_fake_news_model",    
    repo_id="maheen21/fake-news-bert",        
    repo_type="model"
)
