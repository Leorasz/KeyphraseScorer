import torch
import os
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
from docx import Document
import json

device = (
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"  # noqa: E501
)
print(f"Device is {device}")

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
assert models, "No models listed in models.txt"

# Get test files
entries = os.listdir(".")
subdirectories = set(
    [entry for entry in entries if os.path.isdir(os.path.join(".", entry))]
)
known = set(["docs", "ExampleResults", "ExampleHTMLFiles", ".git"])
assert len(known) > 4, "No directory for input text found."
assert len(known) < 6, "Too many other directories, program doesn't know which one to use." # noqa: E501
directory_name = list((subdirectories - known))[0]
directory_path = os.path.join(".", directory_name)

texts = []
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
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
assert texts, "Directory was empty, nothing to input"

# Get keyphrases
with open("keyphrases.txt", "r") as file:
    keyphrases = [keyphrase.strip() for keyphrase in file.readlines()]
assert keyphrases, "No keyphrases in keyphrases.txt"

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
