from openai import OpenAI

client = OpenAI(
    api_key="sk-34dff1924717470eae37eacb1fdde548",
    base_url="https://api.deepseek.com/v1"
)

res = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role":"user","content":"我需要一些推荐的游戏，适合和家人一起玩"}],
    temperature=0.7
)
print(res.choices[0].message.content)