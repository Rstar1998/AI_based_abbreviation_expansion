import pandas as pd

df = pd.read_json("hf://datasets/AGBonnet/augmented-clinical-notes/augmented_notes_30K.jsonl", lines=True)
print(df.head())