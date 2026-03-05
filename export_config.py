
from pathlib import Path
model_home = Path('~/.cache/modelscope/hub/models/iic').expanduser()

# [源模型路径] 官方下载好的 SafeTensors 模型文件夹
MODEL_DIR =  model_home / 'SenseVoiceSmall'

# [导出目标路径] 转换后的模型汇总目录
EXPORT_DIR = Path(r'./model')
