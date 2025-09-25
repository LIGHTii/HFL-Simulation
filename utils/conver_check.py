class ConvergenceChecker:
    """
    根据我们研究报告中的模块化设计，此类封装了提前终止的逻辑。
    它监控验证损失和准确率，当损失和准确率都进入平台期时触发停止信号。
    判断收敛的对象为每轮epoch结束后，由所有EH模型在云端聚合得到的最终全局模型。
    """
    def __init__(self, patience=4, min_delta_loss=0.0001, min_delta_acc=0.1):
        """
        初始化收敛检查器。
        Args:
            patience (int): 在验证损失和准确率都没有改善后，需要等待的轮次数。
            min_delta_loss (float): 被认为是"改善"的最小损失下降值。
            min_delta_acc (float): 被认为是"改善"的最小准确率提升值（百分数形式，如0.1表示0.1%）。
        """
        self.patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_acc = min_delta_acc
        self.wait_rounds = 0
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.stopped_epoch = 0
        self.loss_history = []
        self.acc_history = []
        print(f"收敛检查器已初始化: 耐心值={self.patience}, 损失改善阈值={self.min_delta_loss}, 准确率改善阈值={self.min_delta_acc}")
    
    def check(self, current_loss, current_acc_or_epoch=None, epoch=None):
        """
        检查是否满足停止条件。
        Args:
            current_loss (float): 当前轮次的验证损失。
            current_acc_or_epoch (float|int): 当前轮次的验证准确率，或者轮次编号（向后兼容）。
            epoch (int, optional): 当前的轮次编号。
        Returns:
            (bool, str): 一个元组，包含是否应该停止的布尔值和原因说明。
        """
        # 向后兼容：如果只传入两个参数，认为是旧的接口
        if epoch is None:
            # 旧接口：check(current_loss, epoch)
            epoch = current_acc_or_epoch
            # 只基于损失进行检查
            if current_loss < self.best_loss - self.min_delta_loss:
                self.best_loss = current_loss
                self.wait_rounds = min(0, self.wait_rounds-1)
                return False, f"验证损失改善至 {self.best_loss:.4f}."
            else:
                self.wait_rounds += 1
                if self.wait_rounds >= self.patience:
                    self.stopped_epoch = epoch
                    return True, f"验证损失连续 {self.patience} 轮未改善，触发提前停止。"
                else:
                    remaining = self.patience - self.wait_rounds
                    return False, f"验证损失未改善，等待 {remaining} / {self.patience} 轮..."
        
        # 新接口：check(current_loss, current_acc, epoch)
        current_acc = current_acc_or_epoch
        
        # 记录历史数据
        self.loss_history.append(current_loss)
        self.acc_history.append(current_acc)
        
        # 检查损失是否改善
        loss_improved = current_loss < self.best_loss - self.min_delta_loss
        # 检查准确率是否改善
        acc_improved = current_acc > self.best_acc + self.min_delta_acc
        
        # 如果损失或准确率有任何一个改善，则重置等待轮次
        if loss_improved or acc_improved:
            if loss_improved:
                self.best_loss = current_loss
            if acc_improved:
                self.best_acc = current_acc
            self.wait_rounds = 0
            
            improvement_info = []
            if loss_improved:
                improvement_info.append(f"损失改善至 {self.best_loss:.4f}")
            if acc_improved:
                improvement_info.append(f"准确率改善至 {self.best_acc:.2f}%")
            
            return False, f"{', '.join(improvement_info)}."
        else:
            self.wait_rounds += 1
            if self.wait_rounds >= self.patience:
                self.stopped_epoch = epoch
                return True, f"损失和准确率连续 {self.patience} 轮均未改善，触发提前停止。"
            else:
                remaining = self.patience - self.wait_rounds
                return False, f"损失和准确率均未改善，等待 {remaining} / {self.patience} 轮..."
