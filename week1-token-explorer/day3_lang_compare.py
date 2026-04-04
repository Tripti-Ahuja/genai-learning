import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

pairs = [
    ("I am a data analyst", "मैं एक डेटा एनालिस्ट हूँ"),
    ("Show me total revenue by quarter", "मुझे तिमाही के अनुसार कुल राजस्व दिखाओ"),
    ("What is the average deal size in Salesforce", "सेल्सफोर्स में औसत डील साइज क्या है"),
]

print(f"{'English':<45} {'EN Tok':<8} {'Hindi':<45} {'HI Tok':<8} {'Ratio'}")
print("-" * 120)

for eng, hin in pairs:
    eng_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1,
        messages=[{"role": "user", "content": eng}]
    )
    hin_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1,
        messages=[{"role": "user", "content": hin}]
    )

    en_tok = eng_resp.usage.input_tokens
    hi_tok = hin_resp.usage.input_tokens
    ratio = round(hi_tok / en_tok, 1)

    print(f"{eng:<45} {en_tok:<8} {hin:<45} {hi_tok:<8} {ratio}x")