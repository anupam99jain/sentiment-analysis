import os
import pandas as pd

def load_imdb_split(split_dir):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(split_dir, label_type)
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(1 if label_type == "pos" else 0)
    return pd.DataFrame({"review": texts, "label": labels})

train_df = load_imdb_split("/content/aclImdb/train")
test_df  = load_imdb_split("/content/aclImdb/test")

print(train_df.shape, test_df.shape)
print(train_df.head(3))
