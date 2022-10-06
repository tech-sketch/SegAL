import matplotlib.pyplot as plt


def plot_loss(
    loss_train_logs, loss_val_logs, loss_test_logs, save_path="./output/loss.png"
):
    plt.plot(loss_train_logs, label="train", marker="o")
    plt.plot(loss_val_logs, label="val", marker="o")
    plt.plot(loss_test_logs, label="val", marker="o")
    plt.title("Loss per round")
    plt.ylabel("DiceLoss")
    plt.xlabel("round")
    plt.legend(), plt.grid()
    plt.savefig(save_path)
    # plt.show()


def plot_score(
    iou_train_logs, iou_val_logs, iou_test_logs, save_path="./output/score.png"
):
    plt.plot(iou_train_logs, label="train_mIoU", marker="*")
    plt.plot(iou_val_logs, label="val_mIoU", marker="*")
    plt.plot(iou_test_logs, label="test_mIoU", marker="*")
    plt.title("Score per round")
    plt.ylabel("mean IoU")
    plt.xlabel("round")
    plt.legend(), plt.grid()
    plt.savefig(save_path)
    # plt.show()
