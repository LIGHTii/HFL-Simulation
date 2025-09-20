#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 matplotlib 字体问题
"""

import matplotlib.pyplot as plt
import os
import sys

def set_matplotlib_english_fonts():
    """
    配置 matplotlib 使用英文字体，避免中文字体警告
    """
    print("配置 matplotlib 使用英文字体...")
    
    # 重置字体设置为默认英文字体
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': True  # 正确显示负号
    })
    
    print("matplotlib 字体配置完成")

if __name__ == "__main__":
    set_matplotlib_english_fonts()
    print("运行这个脚本会将 matplotlib 配置为使用英文字体")