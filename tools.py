"""
工具函数模块 —— 统一管理 AI 智能体可调用的所有工具。

每个工具由两部分组成：
  1. 一个 Python 函数（实际执行逻辑）
  2. 一条 OpenAI Function Calling 格式的工具定义

新增工具只需：
  - 在下方编写函数
  - 在 TOOLS 列表中添加对应的定义
  - 在 TOOL_MAP 中注册映射
"""

import ast
import operator

# ----------------------------------------------------------------
# 工具函数实现
# ----------------------------------------------------------------


def get_weather(city: str) -> str:
    """模拟天气查询，返回固定的天气信息。"""
    weather_data = {
        "北京": "【模拟】北京今天晴，22°C，微风，空气质量良",
        "上海": "【模拟】上海今天多云，25°C，东南风3级，空气质量优",
        "广州": "【模拟】广州今天阵雨，28°C，南风2级，湿度偏高",
        "深圳": "【模拟】深圳今天晴，29°C，西南风3级，空气质量优",
        "杭州": "【模拟】杭州今天阴，20°C，东北风2级，有短时小雨",
        "成都": "【模拟】成都今天多云，24°C，微风，适合出行",
        "武汉": "【模拟】武汉今天晴，27°C，南风2级，空气质量良",
        "南京": "【模拟】南京今天多云，23°C，东风3级，体感舒适",
    }
    return weather_data.get(city, f"【模拟】{city}今天多云，23°C，微风")


def get_news() -> str:
    """
    获取国内热门新闻（模拟数据）。
    后续可替换为真实的新闻 API 请求，例如：
        import requests
        resp = requests.get("https://newsapi.example.com/top?country=cn")
        return format_news(resp.json())
    """
    news_list = [
        "🔥 发改委：新一轮促消费政策落地，多地将发放消费券",
        "🌧️ 中央气象台发布暴雨蓝色预警，华南多地需防范城市内涝",
        "🚀 我国成功发射一颗高分辨率对地观测卫星，将用于国土普查",
        "📱 国产 AI 大模型通过国家备案，已在多个政务场景试点应用",
        "🏀 CBA 总决赛：广东队主场险胜辽宁，大比分 2:1 领先",
        "💡 工信部：今年 5G 基站将突破 400 万个，覆盖所有地级市",
        "🎓 教育部公布高考改革新方案，多省份将采用「3+1+2」模式",
        "🌍 联合国气候变化大会即将开幕，中方代表将提出中国方案",
    ]

    lines = ["📰 **国内热门新闻速览**\n"]
    for i, item in enumerate(news_list, 1):
        lines.append(f"{i}. {item}")

    return "\n".join(lines)


# 安全的数学运算白名单
_SAFE_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
}


def _safe_eval(expr: str) -> float | int:
    """安全地计算数学表达式，仅允许基本算术运算和括号。"""
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e.msg}")

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _walk(node.operand)
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op(operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的操作: {type(node).__name__}")

    result = _walk(tree)
    return result


def calculate(expression: str) -> str:
    """
    执行数学计算。
    支持：+, -, *, /, //, %, **, 括号。
    示例：'1234 * 5678' → 7006652
    """
    try:
        result = _safe_eval(expression)
        # 整数去小数点
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"计算结果：{expression} = {result}"
    except ValueError as e:
        return f"⚠️ 计算失败：{e}"
    except ZeroDivisionError:
        return "⚠️ 计算失败：除数不能为零"


# ----------------------------------------------------------------
# 工具定义（OpenAI Function Calling 格式）
# ----------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息。当用户询问某地天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州、深圳",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "获取国内最新热门新闻列表。当用户想看新闻、了解时事时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除、幂运算、取余和括号。当用户需要算数、计算时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '1234 * 5678'、'(100 + 200) / 3'、'2 ** 10'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# 工具名称 → 实际可调用函数的映射
TOOL_MAP = {
    "get_weather": get_weather,
    "get_news": get_news,
    "calculate": calculate,
}
