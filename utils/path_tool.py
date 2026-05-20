"""路径工具模块。

这个文件很小，但在初学阶段很值得单独理解：
1. 项目里很多地方都需要“从项目根目录出发去找文件”
2. 如果把绝对路径写死，项目一换机器就容易出错
3. 所以这里统一封装一个“相对项目根目录取绝对路径”的小工具
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 解释：
# - __file__ 是当前文件自己的路径
# - resolve() 会把它变成绝对路径
# - parents[1] 表示往上走两层，得到项目根目录
# 之所以是两层，是因为当前文件在 utils/path_tool.py。


def get_abs_path(relative_path: str) -> str:
    """把项目内的相对路径转换成绝对路径。

    示例：
    get_abs_path("logs")
    -> /你的项目目录/logs

    这样做的好处是：
    - 不需要在别处重复拼路径
    - 日志、数据、配置文件都能统一定位
    """
    return str((PROJECT_ROOT / relative_path).resolve())
