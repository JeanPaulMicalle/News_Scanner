import os
from config.configLoader import load_config
from news_file_creators.times_of_malta.extract_article_json import extract_article_json_ld

config = load_config()
storage_dir = config["times_of_malta_storage"]      
output_dir  = config["json_output_directory"]      
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(storage_dir):
    if not filename.lower().endswith((".html", ".htm")):
        continue

    input_path = os.path.join(storage_dir, filename)
    base_name, _ = os.path.splitext(filename)
    output_filename = f"{base_name}.json"
    output_path = os.path.join(output_dir, output_filename)

    try:
        extract_article_json_ld(input_path, output_path)
    except Exception as e:
        print(f"Error on {filename}: {e}")

print("Finished processing all files.")
