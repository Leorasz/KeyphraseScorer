import os
import json
import logging
import pandas as pd
from torch import cuda
from docx import Document
from bs4 import BeautifulSoup
from transformers import pipeline

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
print(f"Device is {device}")

# Set up error logging
logging.basicConfig(
    filename="error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def logError(error):
    logging.error(error)
    raise ValueError(error)


# Get models
with open("models.txt", "r") as file:
    models = [
        (
            model_name.strip(),
            pipeline(
                "zero-shot-classification",
                model=model_name.strip(),
                device=0 if device != "cpu" else -1,
            ),
        )
        for model_name in file.readlines()
    ]
if not models:
    logError("No models found in models.txt")

# Get test files
entries = os.listdir(".")
subdirectories = set(
    [entry for entry in entries if os.path.isdir(os.path.join(".", entry))]
)
known = set(["docs", "ExampleResults", "ExampleHTMLFiles", ".git"])
if len(subdirectories) < 5:
    logError("No data directory found")
if len(subdirectories) > 5:
    logError("Too many directories, don't know which one to use")
directory_name = list((subdirectories - known))[0]
directory_path = os.path.join(".", directory_name)

texts = []
for dirpath, _, filenames in os.walk(directory_path):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        if filename.endswith(".docx"):
            doc = Document(file_path)
            text = [paragraph.text for paragraph in doc.paragraphs]
            texts.append((filename[:-5], "\n".join(text)))
            continue
        with open(file_path, "r", encoding="utf-8") as file:
            if filename.endswith(".html"):
                html_content = file.read()
                soup = BeautifulSoup(html_content, "html.parser")
                article_text = " ".join(
                    [
                        element.get_text(strip=True)
                        for element in soup.find_all(
                            ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
                        )
                    ]
                )
                texts.append((filename[:-5], article_text))
            elif filename.endswith(".txt"):
                texts.append((filename[:-4], file.read().strip()))
            elif filename.endswith(".md"):
                texts.append((filename[:-3], file.read().strip()))
            else:
                raise ValueError(
                    f"Filetype not supported for {filename}, only .docx, .html, .txt and .md are supported"  # noqa: E501
                )
if not texts:
    logError("No input data found in the given directory")

# Get keyphrases
with open("keyphrases.txt", "r") as file:
    keyphrases = [keyphrase.strip() for keyphrase in file.readlines()]
if not keyphrases:
    logError("No keyphrases foundin keyphrases.txt")

# Generate results
os.makedirs("Results", exist_ok=True)
for filename, text in texts:
    data = {}
    for model_name, model in models:
        data[model_name] = {}
        for keyphrase in keyphrases:
            result = model(text, [keyphrase])
            score = result["scores"][0]
            data[model_name][keyphrase] = score
    df = pd.DataFrame(data)
    data_dict = df.to_dict(orient="index")
    with open("Results/" + filename + ".json", "w") as file:
        json.dump(data_dict, file, indent=4)
