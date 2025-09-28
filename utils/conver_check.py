class ConvergenceChecker:
    """
    根据我们研究报告中的模块化设计，此类封装了提前终止的逻辑。
    它监控验证损失和准确率，当损失降到指定阈值以下且准确率达到指定阈值以上时触发停止信号。
    判断收敛的对象为每轮epoch结束后，由所有EH模型在云端聚合得到的最终全局模型。
    """
    def __init__(self, patience=4, loss_threshold=0.1, acc_threshold=95.0):
        """
        初始化收敛检查器。
        Args:
            patience (int): 达到收敛条件后，需要持续满足的轮次数。
            loss_threshold (float): 损失收敛阈值，当损失低于此值时认为损失已收敛。
            acc_threshold (float): 准确率收敛阈值，当准确率高于此值时认为准确率已收敛。
        """
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.convergence_count = 0  # 连续满足收敛条件的轮次数
        self.stopped_epoch = 0
        self.loss_history = []
        self.acc_history = []
        print(f"收敛检查器已初始化: 耐心值={self.patience}, 损失收敛阈值={self.loss_threshold}, 准确率收敛阈值={self.acc_threshold}%")
    
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
            # 只基于损失阈值进行检查
            self.loss_history.append(current_loss)
            
            if current_loss <= self.loss_threshold:
                self.convergence_count += 1
                if self.convergence_count >= self.patience:
                    self.stopped_epoch = epoch
                    return True, f"损失连续 {self.patience} 轮低于阈值 {self.loss_threshold}，触发收敛停止。"
                else:
                    remaining = self.patience - self.convergence_count
                    return False, f"损失已达到阈值，持续 {self.convergence_count}/{self.patience} 轮，还需 {remaining} 轮确认收敛。"
            else:
                self.convergence_count = 0
                return False, f"损失 {current_loss:.4f} 高于阈值 {self.loss_threshold}，未收敛。"
        
        # 新接口：check(current_loss, current_acc, epoch)
        current_acc = current_acc_or_epoch
        
        # 记录历史数据
        self.loss_history.append(current_loss)
        self.acc_history.append(current_acc)
        
        # 检查是否同时满足损失和准确率阈值
        loss_converged = current_loss <= self.loss_threshold
        acc_converged = current_acc >= self.acc_threshold
        
        if loss_converged and acc_converged:
            self.convergence_count += 1
            if self.convergence_count >= self.patience:
                self.stopped_epoch = epoch
                return True, f"损失({current_loss:.4f}≤{self.loss_threshold})和准确率({current_acc:.2f}%≥{self.acc_threshold}%)连续 {self.patience} 轮满足收敛条件，触发停止。"
            else:
                remaining = self.patience - self.convergence_count
                return False, f"收敛条件已满足，持续 {self.convergence_count}/{self.patience} 轮，还需 {remaining} 轮确认收敛。"
        else:
            self.convergence_count = 0
            status_info = []
            if not loss_converged:
                status_info.append(f"损失 {current_loss:.4f} > {self.loss_threshold}")
            if not acc_converged:
                status_info.append(f"准确率 {current_acc:.2f}% < {self.acc_threshold}%")
            
            return False, f"未满足收敛条件: {', '.join(status_info)}。"
