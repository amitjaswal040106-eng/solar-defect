import pandas as pd
from data import load_elpv, get_splits

_, labels, df = load_elpv()
tr, va, te = get_splits(labels)
test = df.iloc[te]
bad = pd.read_csv("outputs/resnet18_seed0_misclassified.csv")

out = pd.DataFrame({
    "test_n": test["proba"].round(2).value_counts(),
    "errors": bad["proba"].round(2).value_counts(),
}).fillna(0)
out["error_rate"] = (out["errors"] / out["test_n"]).round(3)
print(out.sort_index())