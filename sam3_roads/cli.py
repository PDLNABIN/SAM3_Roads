from __future__ import annotations

import argparse

from sam3_roads.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAM3 LoRA road segmentation models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training from a TOML config.")
    train_parser.add_argument("--config", required=True, help="Path to a SAM3 roads TOML config.")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint on the test split.")
    evaluate_parser.add_argument("--config", required=True, help="Path to a SAM3 roads TOML config.")
    evaluate_parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    assets_parser = subparsers.add_parser("export-portfolio-assets", help="Export visual case-study assets.")
    assets_parser.add_argument("--config", required=True, help="Path to a SAM3 roads TOML config.")
    assets_parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    assets_parser.add_argument("--output", default="portfolio/assets", help="Asset output directory.")

    args = parser.parse_args()
    if args.command == "train":
        from sam3_roads.train import run_training

        config = load_config(args.config)
        result = run_training(config)
        print(f"Best Val IoU: {result['best_val_iou']:.4f}")
        print(f"Checkpoint: {result['checkpoint_path']}")
        print(f"History: {result['history_path']}")
        print(f"Curves: {result['curves_path']}")
        print(f"Preview: {result['preview_path']}")
        print(f"Metrics: {result['metrics_path']}")
    elif args.command == "evaluate":
        from pathlib import Path
        from sam3_roads.train import evaluate_checkpoint

        result = evaluate_checkpoint(load_config(args.config), Path(args.checkpoint))
        print(f"Evaluation: {result}")
    elif args.command == "export-portfolio-assets":
        from pathlib import Path
        from sam3_roads.visualize import export_portfolio_assets

        export_portfolio_assets(load_config(args.config), Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()
