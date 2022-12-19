from typing import List

import matplotlib.pyplot as plt


def plot_loss(
    full_result: dict,
    random_result: List[dict],
    strategy_result: List[dict],
    save_path: str = "./output/loss.png",
) -> None:
    """Plot loss graph.

    Args:
        full_result (List[dict]): Supervised result.
        random_result (List[dict]): Random result.
        strategy_result (List[dict]): Strategy result.
        save_path (str, optional): Path to save. Defaults to "./output/loss.png".
    """

    random_loss = [log["dice_loss"] for log in random_result]
    strategy_loss = [log["dice_loss"] for log in strategy_result]
    full_loss = [full_result["dice_loss"]] * len(random_loss)

    plt.plot(full_loss, label="Supervised", linestyle="dashed")
    plt.plot(random_loss, label="Random", marker="o")
    plt.plot(strategy_loss, label="Strategy", marker="o")
    plt.title("Loss per round")
    plt.ylabel("DiceLoss")
    plt.xlabel("round")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)


def plot_score(
    full_result: dict,
    random_result: List[dict],
    strategy_result: List[dict],
    save_path: str = "./output/score.png",
) -> None:
    """Plot score graph.

    Args:
        full_result (List[dict]): Supervised result.
        random_result (List[dict]): Random result.
        strategy_result (List[dict]): Strategy result.
        save_path (str, optional): Path to save. Defaults to "./output/loss.png".
    """
    random_score = [log["iou_score"] for log in random_result]
    strategy_score = [log["iou_score"] for log in strategy_result]
    full_score = [full_result["iou_score"]] * len(random_score)

    plt.plot(full_score, label="Supervised", linestyle="dashed")
    plt.plot(random_score, label="Random", marker="o")
    plt.plot(strategy_score, label="Strategy", marker="o")
    plt.title("Score per round")
    plt.ylabel("mean IoU")
    plt.xlabel("round")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
