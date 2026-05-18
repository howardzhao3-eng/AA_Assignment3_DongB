"""
Demo: 格式化游戏推荐输出
=========================
独立脚本，不修改原有代码。使用自定义 prompt 让 LLM 按资深游戏编辑的风格输出推荐。

用法:
  python experiments/demo_formatted_output.py
  python experiments/demo_formatted_output.py --query "我想找一款 cozy 的卡牌游戏"
  python experiments/demo_formatted_output.py --exp E2_L1_D2_S2 --query "适合和朋友联机的生存游戏"
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 raglooker 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from raglooker.recommender_factory import create_search_engine
from raglooker.config import get_experiment_config, FINAL_CONFIG


def build_context(matches: list[dict]) -> str:
    """从 search() 返回的 matches 构建 LLM 上下文。"""
    context_list = []
    for i, match in enumerate(matches[:5], 1):
        name = match.get("name", "未知游戏")
        desc = match.get("short_description", "暂无描述")
        genres = ", ".join(match.get("genres", [])[:3]) or "未知类型"
        tags_list = match.get("tags", [])[:6]
        tags_str = ", ".join(tags_list) if tags_list else "无标签"
        price = match.get("price", "未知")
        release = match.get("release_date", "未知")
        platforms = match.get("platforms", {})
        platform_str = "/".join(
            p for p, enabled in platforms.items() if enabled
        ) or "未知平台"

        context_list.append(
            f"[游戏 {i}]\n"
            f"名称：{name}\n"
            f"简介：{desc}\n"
            f"类型：{genres}\n"
            f"标签：{tags_str}\n"
            f"价格：{price}\n"
            f"发行日期：{release}\n"
            f"平台：{platform_str}\n"
        )
    return "\n---\n".join(context_list)


def generate_curated_recommendation(
    query: str,
    matches: list[dict],
    llm_model: str = "gemma2:2b",
) -> str:
    """
    使用自定义 prompt 生成资深游戏编辑风格的推荐。
    不修改 recommender.py 的原有逻辑。
    """
    if not matches:
        return "没有找到匹配的游戏。"

    context_text = build_context(matches)

    prompt = (
        f"你是一位资深游戏编辑，请根据用户的需求推荐最匹配的 Steam 游戏。\n\n"
        f"用户需求：'{query}'\n\n"
        f"候选游戏数据：\n{context_text}\n\n"
        f"请从候选游戏中选出最匹配的 1~3 款，严格按以下格式输出：\n\n"
        f"🎮 推荐游戏：《游戏名称》\n"
        f"• 类型 / 平台 / 发售年份 / 评分\n"
        f"• 推荐理由：3~4个要点，每个聚焦一个维度（玩法、叙事、美术、创新、情感等），用具体细节说明\n"
        f"• 适合人群：明确写出哪几类玩家会喜欢\n"
        f"• 类似游戏：2~3款并简述相似处\n"
        f"• 点睛句：用一句话唤起共鸣或好奇心\n\n"
        f"排版要求：\n"
        f"- 恰当使用 emoji 但勿滥用\n"
        f"- 各板块之间空行分隔\n"
        f"- 总篇幅控制在 300 字左右\n"
        f"- 语气专业热情，有感染力但不浮夸\n\n"
        f"注意：如果候选数据中缺少某些信息（如发售年份、评分），可以合理推断或省略该字段。"
    )

    try:
        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位资深游戏编辑，对 Steam 游戏了如指掌。"
                        "你的推荐风格专业、热情、有感染力，但不浮夸。"
                        "你擅长用精炼的语言和具体的细节打动读者。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"生成推荐时出错：{e}"


def main():
    parser = argparse.ArgumentParser(
        description="Demo: 格式化游戏推荐输出（资深编辑风格）"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="游戏需求描述（如果不提供，则交互式输入）",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=FINAL_CONFIG,
        help=f"实验配置 ID（默认: {FINAL_CONFIG}）",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM 模型名称（默认使用实验配置中的模型）",
    )
    args = parser.parse_args()

    # 获取查询
    query = args.query
    if not query:
        query = input("请输入游戏需求描述：\n> ").strip()
    if not query:
        print("查询不能为空。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🔍 查询: \"{query}\"")
    print(f"⚙️  配置: {args.exp}")
    print(f"{'='*60}\n")

    # 创建搜索引擎
    print("正在加载游戏数据并构建索引...")
    config = get_experiment_config(args.exp)
    engine = create_search_engine(config)
    print("索引完成。\n")

    # 执行搜索
    print("正在搜索匹配游戏...")
    result = engine.search(query)
    matches = result.get("matches", [])
    meta = result.get("meta", {})

    print(f"找到 {len(matches)} 个候选游戏")
    print(f"检索模式: {meta.get('retrieval_mode', 'N/A')}")
    print(f"嵌入模型: {meta.get('embed_model', 'N/A')}")
    print(f"LLM 模型: {meta.get('llm_model', 'N/A')}")
    timing = meta.get("timing_ms", {})
    print(f"耗时: {timing.get('total_ms', 'N/A')}ms\n")

    # 生成格式化推荐
    llm_model = args.llm or config["llm_model"]["name"]
    print("正在生成推荐...")
    answer = generate_curated_recommendation(query, matches, llm_model)

    # 输出结果
    print(f"\n{'='*60}")
    print("📝 推荐结果")
    print(f"{'='*60}\n")
    print(answer)
    print(f"\n{'='*60}")
    print("✨ 推荐完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
