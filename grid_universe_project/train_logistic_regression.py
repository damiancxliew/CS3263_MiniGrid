from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils import generate_sklearn_loader_snippet

ROOT = Path(__file__).resolve().parent
CIPHERTEXT_PATH = ROOT / "data" / "cipher_objective.csv"
df = pd.read_csv(CIPHERTEXT_PATH)
X = df["text"].astype(str)
y = df["class"].astype(str)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=1000,

    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=3.0,
        n_jobs=1,
    )),
])

pipeline.fit(X, y)

code = generate_sklearn_loader_snippet(pipeline, compression="lzma")
print(code)
