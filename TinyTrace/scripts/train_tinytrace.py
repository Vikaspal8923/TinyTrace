from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinytrace import JsonTinyTraceDataset, SyntheticTinyTraceDataset, TinyTraceConfig, TinyTraceModel, tinytrace_collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--dataset-json", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="TinyTrace/outputs")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = TinyTraceConfig()
    dataset = (
        JsonTinyTraceDataset(args.dataset_json, config=config)
        if args.dataset_json
        else SyntheticTinyTraceDataset(config=config, size=args.dataset_size)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=tinytrace_collate_fn)

    model = TinyTraceModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        steps = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            frame_times = batch["frame_times"].to(device)
            token_ids = batch["token_ids"].to(device)
            label_types = batch["label_types"].to(device)

            optimizer.zero_grad()
            output = model(frames, frame_times, token_ids, labels=token_ids, label_types=label_types)
            if output.loss is None:
                continue
            output.loss.backward()
            optimizer.step()

            running_loss += output.loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, output_dir / "tinytrace.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
