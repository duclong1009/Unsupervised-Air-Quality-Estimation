import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--seed", default=52, type=int, help="Seed")
    parser.add_argument("--target_station", default=14, type=int)
    parser.add_argument(
        "--train_station", default=[int(i) for i in range(28)], type=list,
    )
    parser.add_argument("--input_dim", default=1, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--load_model", default=False)
    parser.add_argument("--output_stdgi", default=60, type=int)
    parser.add_argument("--checkpoint_file", default="./checkpoint/stdgi/", type=str)
    parser.add_argument("--visualize_dir", default="./output/visualize/", type=str)
    parser.add_argument("--topology_construction", default="distance", type=str)
    parser.add_argument("--path_model", default="", type=str)
    return parser.parse_args()