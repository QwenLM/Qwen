# Auto Comments 
本文档介绍Auto Comments，这是一个利用Qwen模型为代码文件自动生成注释的使用案例。

# 使用方法
您可以直接执行如下命令，为提供的代码文件生成注释：
```
python auto_comments.py --path 'path of file or folder'
```

参数：
- path：文件路径。可以是文件（目前支持python代码文件），也可以是文件夹（会扫描文件夹下所有python代码文件）
- regenerate：重新生成。默认False，如果针对同一文件需要重新生成注释，请设置为True

# 使用样例
- 执行：python auto_comments.py --path test_file.py
- test_file.py 内容为：
```
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```

- 输出：test_file_comments.py(包含注释的代码文件)，文件内容如下：
```
# 导入需要的库
import numpy as np
import pandas as pd
import seaborn as sns

# 设置 Seaborn 的主题风格为白色网格
sns.set_theme(style="whitegrid")

# 生成随机数
rs = np.random.RandomState(365)

# 生成 365 行 4 列的随机数，并按行累加
values = rs.randn(365, 4).cumsum(axis=0)

# 生成日期
dates = pd.date_range("1 1 2016", periods=365, freq="D")

# 将随机数和日期组合成 DataFrame
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])

# 对 DataFrame 进行 7 天滑动平均
data = data.rolling(7).mean()

# 使用 Seaborn 绘制折线图
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```
