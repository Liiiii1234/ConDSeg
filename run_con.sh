#!/bin/bash

#SBATCH --job-name=cong      # 任务名
#SBATCH --partition=aiaca800          # GPU 队列
#SBATCH --qos=2a800                   # QoS 配置
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks-per-node=1           # 每节点进程数
#SBATCH --cpus-per-task=8             # 申请 8 核，保证数据加载流畅
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU (与 single GPU 设定一致)
#SBATCH --mem=32G                     # 申请 32G 内存
#SBATCH --output=run_cong_%j.log   # 正常输出的日志文件
#SBATCH --error=run_cong_%j.err    # 报错信息的日志文件
#SBATCH --mail-type=END,FAIL          # 结束或失败时发邮件
#SBATCH --mail-user=Jianuo.Li23@student.xjtlu.edu.cn

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 1. 进入作业提交时所在的目录
cd $SLURM_SUBMIT_DIR

# 2. 清理默认模块，并加载我们刚才为 I-MedSAM 匹配的 CUDA 11.3.1
module purge
module load cuda/12.1.0-gcc-8.5.0-uz47zxn

# 3. 激活 Conda 虚拟环境
# 注意：确认你的 conda 路径，如果是 miniconda 请改为 ~/miniconda3/etc/...
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate imedsam

echo "========== Starting ConDSeg Stage 1 Training =========="
python train_stage1.py
echo "========== Stage 1 Finished Successfully =========="

# 2. 运行第二阶段训练
# （注意：根据该项目的 README，你可能需要在这里加上参数，把第一阶段生成的权重路径传给它，或者直接在 train.py 文件里改好路径）
echo "========== Starting ConDSeg Stage 2 Training =========="
python train.py
echo "========== All Training Finished =========="