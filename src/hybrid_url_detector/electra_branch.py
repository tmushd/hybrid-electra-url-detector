from __future__ import annotations

import os
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hybrid_url_detector.metrics import compute_binary_metrics, MetricResult


os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from hybrid_url_detector.config import RANDOM_SEED


class UrlTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, url_col: str = "url", label_col: str = "label", max_length: int = 128):
        self.urls = df[url_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.urls[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.labels[idx]
        return enc


@dataclass(frozen=True)
class ElectraTrainResult:
    model_dir: Path
    val_metrics: MetricResult


def train_electra_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: Path,
    model_name: str = "google/electra-small-discriminator",
    url_col: str = "url",
    label_col: str = "label",
    max_length: int = 128,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_train_samples: Optional[int] = None,
    fp16: bool = False,
) -> ElectraTrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if max_train_samples is not None:
        train_df = train_df.sample(n=min(len(train_df), max_train_samples), random_state=RANDOM_SEED).reset_index(drop=True)

    train_ds = UrlTextDataset(train_df, tokenizer, url_col=url_col, label_col=label_col, max_length=max_length)
    val_ds = UrlTextDataset(val_df, tokenizer, url_col=url_col, label_col=label_col, max_length=max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ta_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        report_to=[],
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "seed" in ta_params:
        ta_kwargs["seed"] = RANDOM_SEED
    if "data_seed" in ta_params:
        ta_kwargs["data_seed"] = RANDOM_SEED
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        # Fallback: disable eval if the API doesn't support eval strategy
        ta_kwargs["do_eval"] = True

    args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    # Save best model
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Validation metrics (standard deterministic pass)
    val_logits = trainer.predict(val_ds).predictions
    val_proba = torch.softmax(torch.tensor(val_logits), dim=-1)[:, 1].numpy()
    val_metrics = compute_binary_metrics(val_df[label_col].to_numpy(), val_proba)

    return ElectraTrainResult(model_dir=out_dir, val_metrics=val_metrics)


@dataclass(frozen=True)
class ElectraLoaded:
    tokenizer: any
    model: any
    device: torch.device


def load_electra(model_dir: Path, device: str | None = None) -> ElectraLoaded:
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    model.to(dev)
    return ElectraLoaded(tokenizer=tok, model=model, device=dev)


def predict_electra_proba(loaded: ElectraLoaded, urls: list[str], max_length: int = 128, batch_size: int = 32) -> np.ndarray:
    loaded.model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            enc = loaded.tokenizer(
                batch_urls,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(loaded.device)
            logits = loaded.model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.asarray(probs, dtype=np.float32)
