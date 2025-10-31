import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    epochs = []
    val_loss, val_l1, val_kl = [], [], []
    train_loss, train_l1, train_kl = [], [], []
    
    with open(file_path, 'r') as f:
        content = f.readlines()
    
    epoch = None
    for i in range(1000, len(content)):
        line = content[i].strip()
        
        epoch_match = re.match(r"Epoch (\d+)", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
            continue
        
        val_match = re.match(r"Val loss:\s+([\d\.]+)", line)
        if val_match:
            val_loss.append(float(val_match.group(1)))
            loss_line = content[i + 1].strip()
            loss_match = re.match(r"l1_action:\s+([\d\.]+)\s+kl:\s+([\d\.]+)\s+loss:\s+([\d\.]+)", loss_line)
            if loss_match:
                val_l1.append(float(loss_match.group(1)))
                val_kl.append(float(loss_match.group(2)))
            continue
        
        train_match = re.match(r"Train loss:\s+([\d\.]+)", line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
            loss_line = content[i + 1].strip()
            loss_match = re.match(r"l1_action:\s+([\d\.]+)\s+kl:\s+([\d\.]+)\s+loss:\s+([\d\.]+)", loss_line)
            if loss_match:
                train_l1.append(float(loss_match.group(1)))
                train_kl.append(float(loss_match.group(2)))
            continue
    
    return epochs, val_loss, val_l1, val_kl, train_loss, train_l1, train_kl

def plot_losses(epochs, val_losses, train_losses, labels):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(len(labels)):  # 确保索引是从 `labels` 取值
        axes[0, i].plot(epochs, val_losses[i], label=f'Val {labels[i]}')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].set_title(f'Validation {labels[i]}')
        axes[0, i].legend()
        axes[0, i].grid()
        
        # axes[1, i].plot(epochs, train_losses[i], label=f'Train {labels[i]}')
        # axes[1, i].set_xlabel('Epoch')
        # axes[1, i].set_ylabel('Loss')
        # axes[1, i].set_title(f'Training {labels[i]}')
        # axes[1, i].legend()
        # axes[1, i].grid()
    
    plt.tight_layout()
    plt.show()


# 替换成你的log文件路径
file_path = 'log/TAMAC/sim_stack_cube_scripted/3.17-FV.log'


# 解析log文件
epochs, val_loss, val_l1, val_kl, train_loss, train_l1, train_kl = parse_log(file_path)

epochs, val_loss, val_l1, val_kl, train_loss, train_l1, train_kl = parse_log(file_path)

# 绘制一整幅图，上面是Validation Loss，下面是Training Loss
plot_losses(epochs, [val_loss, val_l1, val_kl], [train_loss, train_l1, train_kl], ['Loss', 'L1', 'KL'])

