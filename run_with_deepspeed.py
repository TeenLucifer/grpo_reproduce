#!/usr/bin/env python3
"""
DeepSpeed集成测试启动脚本
用于验证DeepSpeed配置和训练效果
"""

import subprocess
import sys
import os

def main():
    """主函数：启动带有DeepSpeed的训练"""
    print("=== DeepSpeed集成测试启动 ===")
    print("正在检查DeepSpeed配置...")
    
    # 检查配置文件是否存在
    if not os.path.exists("./deepspeed.json"):
        print("错误：找不到deepspeed.json配置文件")
        sys.exit(1)
    
    print("DeepSpeed配置文件已找到")
    print("启动参数：")
    print("  - DeepSpeed: 启用")
    print("  - 批次大小: 128 (micro_batch: 16)")
    print("  - ZeRO阶段: 2")
    print("  - FP16: 启用")
    print("  - CPU优化器卸载: 启用")
    
    try:
        # 启动主训练脚本
        print("\n开始训练...")
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("训练完成！")
            print("\n训练输出：")
            print(result.stdout)
        else:
            print("训练失败！")
            print("\n错误输出：")
            print(result.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"启动训练时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()