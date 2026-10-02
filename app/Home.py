from __future__ import annotations
import base64
import io
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ECG-XAI Research Application ", page_icon="❤", layout="wide"
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "app" else HERE
for d in ["assets", "models", "reports", "experiments"]:
  (ROOT / d).mkdir(parents=True, exist_ok=True)

BG = ROOT / ("../assets/ecg_hospital_3d.jpg")
if BG.exists():
  b64 = base64.b64encode(BG.read_bytes()).decode()
  background = f"background-image:linear-gradient(rgba(5,18,34,.86),rgba(5,18,34,.93)),url('data:image/jpeg;base64,{b64}');"
else:
  background = "background:linear-gradient(135deg,#061426,#10334a,#061426);"

st.markdown(
    f"""<style>.stApp{{{background}background-size:cover;background-position:center;background-attachment:fixed;}}
[data-testid='stSidebar']{{background:#fff;}} [data-testid='stSidebar'] *{{color:#111827!important;text-shadow:none!important;}}
.stMain p,.stMain h1,.stMain h2,.stMain h3,.stMain li{{color:#f3f7fb;text-shadow:0 1px 3px #000;}}
div[data-testid='stMetric']{{background:rgba(10,35,55,.88);border:1px solid #2dd4bf66;padding:12px;border-radius:12px;}}
div[data-testid='stMetric'] *{{color:white!important;}}</style>""",
    unsafe_allow_html=True,
)


def load_csv(f):
  f.seek(0)
  df = pd.read_csv(f, low_memory=False)
  df.columns = [str(c).strip() for c in df.columns]
  if df.empty:
    raise ValueError("CSV is empty.")
  if df.columns.duplicated().any():
    raise ValueError("Duplicate column names detected.")
  return df


def xy(df, target):
  if target not in df:
    raise ValueError("Target column missing.")
  y = df[target].astype("string").str.strip()
  keep = y.notna() & y.ne("")
  y = y[keep].astype(str)
  X = df.loc[keep].drop(columns=[target]).copy()
  X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
  X = X.dropna(axis=1, how="all")
  if X.empty:
    raise ValueError("No usable numeric feature columns.")
  if y.nunique() < 2:
    raise ValueError("At least two target classes are required.")
  return X, y


def make_estimator(name, seed=42):
  if name == "Random Forest":
    return RandomForestClassifier(
        n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=-1
    )
  if name == "Extra Trees":
    return ExtraTreesClassifier(
        n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=-1
    )
  if name == "Gradient Boosting":
    return GradientBoostingClassifier(random_state=seed)
  if name == "Hist Gradient Boosting":
    return HistGradientBoostingClassifier(random_state=seed)
  if name == "Logistic Regression":
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=2500, class_weight="balanced"),
        ),
    ])
  if name == "SVM":
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            SVC(probability=True, class_weight="balanced", random_state=seed),
        ),
    ])
  if name == "MLP":
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            __import__("sklearn").neural_network.MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            ),
        ),
    ])
  raise ValueError(name)


def metrics(y, pred, proba, classes):
  out = {
      "Accuracy": accuracy_score(y, pred),
      "Balanced accuracy": balanced_accuracy_score(y, pred),
      "Precision weighted": precision_score(
          y, pred, average="weighted", zero_division=0
      ),
      "Recall weighted": recall_score(
          y, pred, average="weighted", zero_division=0
      ),
      "F1 weighted": f1_score(y, pred, average="weighted", zero_division=0),
      "F1 macro": f1_score(y, pred, average="macro", zero_division=0),
      "MCC": matthews_corrcoef(y, pred),
      "Cohen kappa": cohen_kappa_score(y, pred),
  }
  if proba is not None:
    try:
      out["ROC-AUC"] = (
          roc_auc_score(y, proba[:, 1], multi_class="raise")
          if len(classes) == 2
          else roc_auc_score(
              y, proba, multi_class="ovr", average="weighted", labels=classes
          )
      )
    except Exception:
      out["ROC-AUC"] = np.nan
    try:
      out["Log loss"] = log_loss(y, proba, labels=classes)
    except Exception:
      out["Log loss"] = np.nan
    if len(classes) == 2:
      try:
        out["Brier score"] = brier_score_loss(
            (pd.Series(y).astype(str) == str(classes[1])).astype(int),
            proba[:, 1],
        )
      except Exception:
        pass
  return out


def train_sklearn(name, Xtr, ytr, Xte, yte):
  imp = SimpleImputer(strategy="median")
  a = pd.DataFrame(imp.fit_transform(Xtr), columns=Xtr.columns)
  b = pd.DataFrame(imp.transform(Xte), columns=Xtr.columns)
  model = make_estimator(name)
  start = time.perf_counter()
  model.fit(a, ytr)
  elapsed = time.perf_counter() - start
  pred = model.predict(b)
  proba = model.predict_proba(b) if hasattr(model, "predict_proba") else None
  classes = np.array(model.classes_) if hasattr(model, "classes_") else np.unique(ytr)
  return {
      "name": name,
      "model": model,
      "imputer": imp,
      "features": list(Xtr.columns),
      "y_test": np.asarray(yte),
      "X_test": b,
      "pred": np.asarray(pred),
      "proba": proba,
      "classes": classes,
      "elapsed": elapsed,
      "metrics": metrics(np.asarray(yte), np.asarray(pred), proba, classes),
      "report": classification_report(yte, pred, zero_division=0),
  }


def train_torch(name, Xtr, ytr, Xte, yte, epochs, batch, lr):
  import torch
  from torch import nn
  from torch.utils.data import DataLoader, TensorDataset

  classes = sorted(pd.Series(ytr).astype(str).unique())
  encode = {c: i for i, c in enumerate(classes)}
  yt = np.array([encode[str(v)] for v in ytr], dtype=np.int64)
  imp = SimpleImputer(strategy="median")
  a = imp.fit_transform(Xtr).astype("float32")
  b = imp.transform(Xte).astype("float32")
  mu = a.mean(axis=0)
  sd = a.std(axis=0)
  sd[sd == 0] = 1
  a = (a - mu) / sd
  b = (b - mu) / sd
  device = "cuda" if torch.cuda.is_available() else "cpu"
  nclass = len(classes)

  class Net(nn.Module):

    def __init__(self):
      super().__init__()
      if name == "1D-CNN":
        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, nclass),
        )
      else:
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(128, nclass)

    def forward(self, x):
      if name == "1D-CNN":
        return self.body(x)
      _, (h, _) = self.rnn(x)
      return self.fc(self.drop(torch.cat((h[-2], h[-1]), dim=1)))

  torch.manual_seed(42)
  model = Net().to(device)
  weights = np.bincount(yt, minlength=nclass)
  cw = len(yt) / (nclass * np.maximum(weights, 1))
  criterion = nn.CrossEntropyLoss(
      weight=torch.tensor(cw, dtype=torch.float32, device=device)
  )
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  ds = TensorDataset(torch.tensor(a[:, None, :]), torch.tensor(yt))
  loader = DataLoader(ds, batch_size=batch, shuffle=True)
  start = time.perf_counter()
  logs = []
  for ep in range(epochs):
    model.train()
    loss_sum = 0
    for xx, yy in loader:
      xx = xx.to(device)
      yy = yy.to(device)
      opt.zero_grad()
      loss = criterion(model(xx), yy)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
      opt.step()
      loss_sum += loss.item() * len(yy)
    logs.append({"epoch": ep + 1, "loss": loss_sum / len(ds)})
  elapsed = time.perf_counter() - start
  model.eval()
  with torch.no_grad():
    logits = model(torch.tensor(b[:, None, :]).to(device))
    prob = torch.softmax(logits, dim=1).cpu().numpy()
  pred_idx = prob.argmax(axis=1)
  pred = np.array([classes[i] for i in pred_idx])
  true = np.asarray(yte).astype(str)
  return {
      "name": name,
      "model": model,
      "imputer": imp,
      "features": list(Xtr.columns),
      "y_test": true,
      "X_test": pd.DataFrame(b, columns=Xtr.columns),
      "pred": pred,
      "proba": prob,
      "classes": np.array(classes),
      "elapsed": elapsed,
      "metrics": metrics(true, pred, prob, np.array(classes)),
      "report": classification_report(true, pred, zero_division=0),
      "logs": pd.DataFrame(logs),
      "torch_mu": mu,
      "torch_sd": sd,
      "device": device,
      "is_torch": True,
      "torch_classes": classes,
  }


def result_card(r):
  m = r["metrics"]
  cols = st.columns(5)
  for c, k in zip(
      cols, ["Accuracy", "Precision weighted", "Recall weighted", "F1 weighted", "ROC-AUC"]
  ):
    v = m.get(k, np.nan)
    c.metric(k, "N/A" if pd.isna(v) else f"{v:.4f}")
  st.caption(
      f"{r['name']} | training time {r['elapsed']:.2f}s | test n={len(r['y_test'])}"
  )


def render_cm(r):
  labs = list(map(str, r["classes"]))
  cm = confusion_matrix(r["y_test"], r["pred"], labels=labs)
  fig, ax = plt.subplots(figsize=(7, 5))
  ConfusionMatrixDisplay(cm, display_labels=labs).plot(
      ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
  )
  ax.set_title("Confusion Matrix")
  fig.tight_layout()
  st.pyplot(fig)
  plt.close(fig)


def csvdata(df):
  return df.to_csv(index=False).encode("utf-8-sig")


if "frames" not in st.session_state:
  st.session_state.frames = {}
if "result" not in st.session_state:
  st.session_state.result = None
if "runs" not in st.session_state:
  st.session_state.runs = []
if "cv" not in st.session_state:
  st.session_state.cv = None
if "compare" not in st.session_state:
  st.session_state.compare = []
if "loaded" not in st.session_state:
  st.session_state.loaded = None

st.sidebar.title("❤️ ECG-XAI Research Application")
pages = [
    "Dashboard",
    "Upload & Validation",
    "Dataset Explorer",
    "Signal Viewer",
    "Train & Compare",
    "Cross-validation",
    "Evaluation",
    "Explainable AI",
    "Prediction",
    "Model Registry",
    "Experiment History",
    "Research Report",
    "About",
]
page = st.sidebar.radio("Navigation", pages)
st.sidebar.divider()
st.sidebar.caption("Upload CSVs separately")
for key in ["Dataset.csv", "train.csv", "test.csv"]:
  f = st.sidebar.file_uploader(key, type=["csv"], key="file_" + key)
  if f:
    try:
      st.session_state.frames[key] = load_csv(f)
      st.sidebar.success(f"{key} loaded")
    except Exception as e:
      st.sidebar.error(str(e))
frames = st.session_state.frames
source = st.sidebar.selectbox(
    "Input mode", ["Dataset.csv (automatic split)", "train.csv + test.csv"]
)
st.title("❤ ECG-XAI Research Baby!")
st.caption(
    "Automated ECG classification, evaluation and explainability — academic"
    " prototype"
)
st.warning(
    "For research purposes only. Not intended for clinical diagnosis, treatment"
    " decisions, or emergency use."
)


def get_data():
  if source.startswith("Dataset"):
    if "Dataset.csv" not in frames:
      raise ValueError("Please upload Dataset.csv.")
    return frames["Dataset.csv"], None
  if "train.csv" not in frames or "test.csv" not in frames:
    raise ValueError("Please upload both train.csv and test.csv.")
  return frames["train.csv"], frames["test.csv"]


if page == "Dashboard":
  st.header("Project overview")
  a, b, c = st.columns(3)
  a.metric("Uploaded files", len(frames))
  b.metric("Saved models", len(list((ROOT / "models").glob("*"))))
  c.metric("Experiments", len(st.session_state.runs))
  if frames:
    for k, d in frames.items():
      st.write(f"**{k}:** {d.shape[0]:,} rows × {d.shape[1]} columns")
  else:
    st.info("To start, upload Dataset.csv or train.csv + test.csv.")
  if st.session_state.result:
    result_card(st.session_state.result)

elif page == "Upload & Validation":
  st.header("Dataset quality checks")
  if not frames:
    st.info("Please upload CSV files.")
  for k, d in frames.items():
    with st.expander(f"{k} — {d.shape}", True):
      st.dataframe(d.head(15), use_container_width=True)
      st.write("Missing by column")
      st.dataframe(d.isna().sum().rename("missing").to_frame())
      st.write("Duplicate rows:", int(d.duplicated().sum()))
      st.write(
          "Numeric infinite values:",
          int(np.isinf(d.select_dtypes(include=np.number)).sum().sum()),
      )
      st.write("Data types")
      st.dataframe(d.dtypes.astype(str).rename("dtype").to_frame())

elif page == "Dataset Explorer":
  if not frames:
    st.info("Please upload CSV files.")
  else:
    k = st.selectbox("Dataset", list(frames))
    d = frames[k]
    st.write("Shape:", d.shape)
    st.dataframe(d, use_container_width=True)
    col = st.selectbox("Column", d.columns)
    if pd.api.types.is_numeric_dtype(d[col]):
      fig, ax = plt.subplots()
      d[col].dropna().hist(ax=ax, bins=40)
      ax.set_title(col)
      st.pyplot(fig)
      plt.close(fig)
      st.write(d[col].describe().to_frame("statistics"))
    else:
      st.bar_chart(d[col].astype(str).value_counts().head(25))
    st.download_button("Download CSV", csvdata(d), f"{k[:-4]}_copy.csv", "text/csv")

elif page == "Signal Viewer":
  if not frames:
    st.info("Please upload CSV files.")
  else:
    k = st.selectbox("Dataset", list(frames))
    d = frames[k]
    numeric = d.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
      st.warning("No numeric signal/features found.")
    else:
      col = st.selectbox("Signal column", numeric)
      vals = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()
      if len(vals):
        max_v = len(vals)
        min_v = min(100, max_v)
        if min_v >= max_v:
            min_v = max(1, max_v - 1)
            
        n = st.slider(
            "Points to display",
            min_value=min_v,
            max_value=max_v,
            value=min(1000, max_v),
        )
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(vals[:n])
        ax.set(
            xlabel="Sample index (sampling rate not supplied)",
            ylabel="Amplitude",
            title=col,
        )
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

elif page == "Train & Compare":
  st.header("Train models")
  try:
    d, testdf = get_data()
  except Exception as e:
    st.info(str(e))
    d = None
  if d is not None:
    target = st.selectbox("Target column", d.columns)
    model_names = st.multiselect(
        "Models to train",
        [
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "Hist Gradient Boosting",
            "Logistic Regression",
            "SVM",
            "MLP",
            "1D-CNN",
            "LSTM",
        ],
        default=["Random Forest", "Extra Trees"],
    )
    if source.startswith("Dataset"):
      test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    else:
      test_size = None
    dl_models = [x for x in model_names if x in ["1D-CNN", "LSTM"]]
    if dl_models:
      epochs = st.slider("Deep learning epochs", 2, 50, 10)
      batch = st.select_slider(
          "Batch size", options=[8, 16, 32, 64, 128], value=32
      )
      lr = st.select_slider(
          "Learning rate", options=[0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001
      )
    else:
      epochs, batch, lr = 5, 32, 0.001
    if st.button("Train selected models", type="primary", disabled=not model_names):
      try:
        X, y = xy(d, target)
        if testdf is None:
          minc = int(y.value_counts().min())
          strat = (
              y
              if minc >= 2
              and int(len(y) * test_size) >= y.nunique()
              and int(len(y) * (1 - test_size)) >= y.nunique()
              else None
          )
          Xtr, Xte, ytr, yte = train_test_split(
              X, y, test_size=test_size, random_state=42, stratify=strat
          )
        else:
          Xtr, ytr = xy(d, target)
          Xte, yte = xy(testdf, target)
          missing = [f for f in Xtr.columns if f not in Xte.columns]
          if missing:
            raise ValueError(f"Test feature missing: {missing}")
          Xte = Xte[Xtr.columns]
          if set(yte) - set(ytr):
            raise ValueError("Test set contains class absent from train set.")
        trained = []
        progress = st.progress(0)
        for i, name in enumerate(model_names):
          with st.spinner(f"Training {name}..."):
            r = (
                train_torch(name, Xtr, ytr, Xte, yte, epochs, batch, lr)
                if name in ["1D-CNN", "LSTM"]
                else train_sklearn(name, Xtr, ytr, Xte, yte)
            )
          r.update({
              "target": target,
              "source": source,
              "train_rows": len(Xtr),
              "test_rows": len(Xte),
              "timestamp": datetime.now().isoformat(timespec="seconds"),
          })
          trained.append(r)
          progress.progress((i + 1) / len(model_names))
        st.session_state.compare = trained
        st.session_state.result = trained[0]
        for r in trained:
          st.session_state.runs.append({
              **{
                  k: r[k]
                  for k in [
                      "name",
                      "target",
                      "source",
                      "train_rows",
                      "test_rows",
                      "timestamp",
                      "elapsed",
                  ]
              },
              **r["metrics"],
          })
        st.success("Selected models trained.")
      except Exception as e:
        st.error(f"Training error: {e}")
    if st.session_state.compare:
      tab = st.tabs([r["name"] for r in st.session_state.compare])
      for t, r in zip(tab, st.session_state.compare):
        with t:
          result_card(r)
          st.dataframe(
              pd.DataFrame([r["metrics"]]).T.rename(columns={0: "value"})
          )
          render_cm(r)
          st.code(r["report"])
          if "logs" in r:
            fig, ax = plt.subplots()
            ax.plot(r["logs"].epoch, r["logs"].loss)
            ax.set(xlabel="Epoch", ylabel="Loss", title="Training loss")
            st.pyplot(fig)
            plt.close(fig)
      compare = pd.DataFrame([{
          "Model": r["name"],
          **r["metrics"],
          "Training seconds": r["elapsed"],
      } for r in st.session_state.compare])
      st.subheader("Model comparison")
      st.dataframe(compare, use_container_width=True)
      st.download_button(
          "Download model comparison",
          csvdata(compare),
          "model_comparison.csv",
          "text/csv",
      )

elif page == "Cross-validation":
  st.header("Stratified K-Fold Cross-validation")
  if "Dataset.csv" not in frames:
    st.info("Please upload Dataset.csv.")
  else:
    d = frames["Dataset.csv"]
    target = st.selectbox("Target", d.columns, key="cv_target")
    name = st.selectbox(
        "Model",
        [
            "Random Forest",
            "Extra Trees",
            "Logistic Regression",
            "SVM",
            "MLP",
        ],
        key="cv_model",
    )
    max_possible_folds = int(d[target].value_counts().min()) if target in d else 2
    max_folds_val = max(2, min(10, max_possible_folds))
    folds = st.slider("Folds", 2, max_folds_val, min(5, max_folds_val))
    if st.button("Run CV"):
      try:
        X, y = xy(d, target)
        if y.value_counts().min() < folds:
          raise ValueError(
              "Smallest class needs at least as many samples as folds."
          )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("estimator", make_estimator(name)),
        ])
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=[
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ],
            n_jobs=1,
        )
        st.session_state.cv = pd.DataFrame({
            k: scores["test_" + k]
            for k in [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ]
        })
      except Exception as e:
        st.error(str(e))
    if st.session_state.cv is not None:
      st.dataframe(st.session_state.cv, use_container_width=True)
      st.dataframe(st.session_state.cv.agg(["mean", "std"]).T)
      st.download_button(
          "Download CV results",
          csvdata(st.session_state.cv),
          "cv_results.csv",
          "text/csv",
      )

elif page == "Evaluation":
  r = st.session_state.result
  if not r:
    st.info("Please train a model first.")
  else:
    result_card(r)
    st.dataframe(pd.DataFrame([r["metrics"]]).T.rename(columns={0: "value"}))
    render_cm(r)
    st.subheader("Classification report")
    st.dataframe(
        pd.DataFrame(
            classification_report(
                r["y_test"], r["pred"], output_dict=True, zero_division=0
            )
        ).T
    )
    if r["proba"] is not None:
      classes = list(map(str, r["classes"]))
      y = np.asarray(r["y_test"]).astype(str)
      prob = r["proba"]
      if len(classes) == 2:
        pos = classes[1]
        yy = (y == pos).astype(int)
        fpr, tpr, _ = roc_curve(yy, prob[:, 1])
        prec, rec, _ = precision_recall_curve(yy, prob[:, 1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC {auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--")
        ax.set(title="ROC curve", xlabel="FPR", ylabel="TPR")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(rec, prec)
        ax.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
        st.pyplot(fig)
        plt.close(fig)
      else:
        try:
          Y = label_binarize(y, classes=classes)
          fig, ax = plt.subplots()
          for i, c in enumerate(classes):
            if Y[:, i].sum() and Y[:, i].sum() < len(Y):
              fpr, tpr, _ = roc_curve(Y[:, i], prob[:, i])
              ax.plot(fpr, tpr, label=f"{c} AUC={auc(fpr, tpr):.2f}")
          ax.plot([0, 1], [0, 1], "--")
          ax.set(title="One-vs-rest ROC", xlabel="FPR", ylabel="TPR")
          ax.legend()
          st.pyplot(fig)
          plt.close(fig)
        except Exception as e:
          st.info(f"Multiclass curve unavailable: {e}")
    pred = pd.DataFrame({"Actual": r["y_test"], "Predicted": r["pred"]})
    if r["proba"] is not None:
      for i, c in enumerate(r["classes"]):
        pred[f"Probability_{c}"] = r["proba"][:, i]
    st.download_button(
        "Download predictions", csvdata(pred), "predictions.csv", "text/csv"
    )

elif page == "Explainable AI":
  r = st.session_state.result
  if not r:
    st.info("Please train a model first.")
  else:
    model = r["model"]
    st.subheader("Feature importance")
    if hasattr(model, "feature_importances_"):
      imp = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps.get("estimator", model.named_steps.get("model")),
        "coef_",
    ):
      imp = np.mean(
          np.abs(
              model.named_steps.get(
                  "estimator", model.named_steps.get("model")
              ).coef_
          ),
          axis=0,
      )
    else:
      imp = None
    if imp is not None:
      fi = pd.DataFrame({"Feature": r["features"], "Importance": imp}).sort_values(
          "Importance", ascending=False
      )
      st.dataframe(fi, use_container_width=True)
      top = fi.head(20).sort_values("Importance")
      fig, ax = plt.subplots(figsize=(9, 5))
      ax.barh(top.Feature, top.Importance)
      ax.set_title("Top feature importance")
      fig.tight_layout()
      st.pyplot(fig)
      plt.close(fig)
      st.download_button(
          "Download importance",
          csvdata(fi),
          "feature_importance.csv",
          "text/csv",
      )

elif page == "Prediction":
  r = st.session_state.result
  if not r:
    st.info("Please train a model first.")
  elif r.get("is_torch"):
    st.info("Single-row prediction requires classical ML models.")
  else:
    st.caption("Provide numeric input matching the exact feature names.")
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(r["features"]):
      vals[f] = cols[i % 3].number_input(f, value=0.0, key="new_" + f)
    if st.button("Predict"):
      x = pd.DataFrame([vals], columns=r["features"])
      x = pd.DataFrame(r["imputer"].transform(x), columns=r["features"])
      pred = r["model"].predict(x)[0]
      st.success(f"Predicted label: {pred}")
      if hasattr(r["model"], "predict_proba"):
        p = r["model"].predict_proba(x)[0]
        st.dataframe(
            pd.DataFrame({"Class": r["model"].classes_, "Probability": p})
            .sort_values("Probability", ascending=False)
        )

elif page == "Model Registry":
  st.header("Model Registry")
  r = st.session_state.result
  if r:
    st.subheader("Save current model")
    filename = st.text_input("Filename", "ecg_research_model.joblib")
    if st.button("Save model"):
      try:
        safe = Path(filename).stem + ".joblib"
        joblib.dump(
            {
                "model": r["model"],
                "imputer": r["imputer"],
                "features": r["features"],
                "target": r["target"],
                "name": r["name"],
            },
            ROOT / "models" / safe,
        )
        st.success(f"Saved models/{safe}")
      except Exception as e:
        st.error(str(e))

elif page == "Experiment History":
  st.header("Experiment tracking")
  if st.session_state.runs:
    df = pd.DataFrame(st.session_state.runs)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download run history", csvdata(df), "experiment_history.csv", "text/csv"
    )

elif page == "Research Report":
  r = st.session_state.result
  if not r:
    st.info("Please train a model before generating a report.")
  else:
    report = (
        f"""ECG-XAI RESEARCH SUMMARY
Generated: {datetime.now().isoformat(timespec='seconds')}
Model: {r['name']}
Target: {r['target']}
Training samples: {r['train_rows']}
Testing samples: {r['test_rows']}

METRICS
"""
        + "\n".join(
            f"{k}: {v:.5f}" for k, v in r["metrics"].items() if not pd.isna(v)
        )
        + f"""

CLASSIFICATION REPORT
{r['report']}
"""
    )
    st.text_area("Report preview", report, height=400)
    st.download_button(
        "Download TXT", report, "ecg_xai_report.txt", "text/plain"
    )

elif page == "About":
  st.header("About this research prototype")
  st.markdown(
      "**Research theme:** Explainable AI for ECG classification and cardiac"
      " abnormality detection."
  )