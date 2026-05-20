"""离线评测脚本。

这个脚本的用途很简单：
- 不启动页面
- 直接在命令行里跑一次检索评测
- 输出当前 Recall@K / MRR / Hit Rate 等指标

适合场景：
- 改了检索逻辑后快速验收
- 面试前截图评测结果
- 做不同方案前后的效果对比
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 运行脚本时，当前工作目录不一定自动把项目根目录放进 sys.path，
# 所以这里手动插入，保证 `from services...` 能成功导入。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.evaluation_service import EvaluationService


def main() -> None:
    """执行一次离线评测并打印结果。"""
    metrics = EvaluationService().evaluate_recall(k=5)
    print("retrieval_metrics=", metrics)


if __name__ == "__main__":
    main()
