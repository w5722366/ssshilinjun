import re
from collections import defaultdict

# 初始化存储 loss 的字典
epoch_losses = defaultdict(list)

# 打开并读取日志文件
with open('mmselfsup-main/work_dir/Starnet-relation-lr=0.01/20241006_151251/20241006_151251.log', 'r') as log_file:
    for line in log_file:
        # 匹配日志中的 Epoch 和 loss 信息
        match = re.search(r'Epoch\(train\) \[(\d+)\]\[\d+/\d+\].* loss: (\d+\.\d+)', line)
        if match:
            epoch = int(match.group(1))  # 提取 epoch 数字
            loss = float(match.group(2))  # 提取 loss 值
            epoch_losses[epoch].append(loss)  # 将 loss 加入对应的 epoch

# 计算每个 epoch 的平均 loss
for epoch, losses in epoch_losses.items():
    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
