import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取环境变量中的 API 密钥
api_key = os.getenv("OPENAI_API_KEY")

# 确保已获取 API 密钥
if not api_key:
    raise ValueError("API 密钥未设置。请检查 .env 文件中的 OPENAI_API_KEY")

# 初始化openai的客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def get_comforting_message(emotion, prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个心理专家，专门安慰人的情绪。"},
            {"role": "user", "content": prompt}
        ],
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].message.content.strip()
    return message


emotion_prompts = {
    "生气": "请给我一段安慰生气情绪的句子，使用中文回答。请告诉他你理解他的愤怒，并且会支持他度过这段时间。你可以举例说明如何理解和支持对方。",
    "开心": "请给我一段安慰开心情绪的句子，使用中文回答。请表达你对他的开心感到高兴，并且愿意和他一起庆祝这个时刻。你可以描述一些庆祝的具体方式。",
    "平静": "请给我一段安慰平静情绪的句子，使用中文回答。请表达你对他现状的支持，并且愿意随时倾听他的心声。你可以详细描述如何提供支持。",
    "悲伤": "请给我一段安慰悲伤情绪的句子，使用中文回答。请告诉他你理解他的痛苦，并且会在他身边陪伴，帮助他度过这段难关。你可以详细描述如何陪伴和支持对方。",
    "惊讶": "请给我一段安慰惊讶情绪的句子，使用中文回答。请告诉他你理解他的惊讶，并且会和他一起面对新的变化。你可以详细描述如何一起面对这些变化。"
}

if __name__ == "__main__":
    emotion = "悲伤"  # 测试用的情绪

    # 获取对应情绪的提示语
    if emotion in emotion_prompts:
        prompt = emotion_prompts[emotion]
        comforting_message = get_comforting_message(emotion, prompt)
        print(f"{emotion}情绪的安慰信息: {comforting_message}")
    else:
        print(f"未找到对应的提示语: {emotion}")
