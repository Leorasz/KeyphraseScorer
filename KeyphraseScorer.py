import torch
import os
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
from docx import Document

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"  # noqa: E501
print(f"Device is {device}")

# Get models
with open("models.txt", "r") as file:
    model_names = [model_name.strip() for model_name in file.readlines()]

models = [
    (
        model_name,
        pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device != "cpu" else -1,
        ),
    )
    for model_name in model_names
]

# Get test files
directory_name = "SampleHTMLFiles"
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

# Get keyphrases
with open("keyphrases.txt", "r") as file:
    keyphrases = [keyphrase.strip() for keyphrase in file.readlines()]

# Generate results
os.makedirs("Results", exist_ok=True)
for filename, text in texts:
    data = {}
    for model_name, model in models:
        data[model_name] = {}
        for (
            keyphrase
        ) in (
            keyphrases
        ):
            score = model(text, [keyphrase])["scores"][0]
            data[model_name][keyphrase] = score
    df = pd.DataFrame(data)
    json_data = df.to_json(orient="index")
    with open("Results/" + filename + ".json", "w") as file:
        file.write(json_data)
