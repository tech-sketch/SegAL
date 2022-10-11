import json
from typing import Any, List

import matplotlib.pyplot as plt


def plot_loss(
    full_result, random_result, strategy_result, save_path="./output/loss.png"
):

    random_loss = [log["dice_loss"] for log in random_result]
    strategy_loss = [log["dice_loss"] for log in strategy_result]
    full_loss = [full_result["dice_loss"]] * len(random_loss)

    plt.plot(full_loss, label="Supervised", linestyle="dashed")
    plt.plot(random_loss, label="Random", marker="o")
    plt.plot(strategy_loss, label="Strategy", marker="o")
    plt.title("Loss per round")
    plt.ylabel("DiceLoss")
    plt.xlabel("round")
    plt.legend(), plt.grid()
    plt.savefig(save_path)
    # plt.show()


def plot_score(
    full_result, random_result, strategy_result, save_path="./output/score.png"
):

    random_score = [log["iou_score"] for log in random_result]
    strategy_score = [log["iou_score"] for log in strategy_result]
    full_score = [full_result["iou_score"]] * len(random_score)

    plt.plot(full_score, label="Supervised", linestyle="dashed")
    plt.plot(random_score, label="Random", marker="o")
    plt.plot(strategy_score, label="Strategy", marker="o")
    plt.title("Score per round")
    plt.ylabel("mean IoU")
    plt.xlabel("round")
    plt.legend(), plt.grid()
    plt.savefig(save_path)
    # plt.show()


def save_json(data: Any, save_path: str) -> None:
    """Save data to json format.
    Args:
        data (Any): The data to store.
        save_path (str): Path to save.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> List[Any]:
    """Read json data
    Args:
        path (str): Path of file to read.
    Returns:
        List[Any]: FTSE entity data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
