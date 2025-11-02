import json, os, httpx, asyncio
from datetime import datetime

MODEL = "qwen2.5:7b-instruct-q4_K_M"
INPUT = "logs/gost_eval.jsonl"
OUT_JSON = "logs/gost_eval_self.jsonl"
OUT_HTML = "logs/gost_eval_self_report.html"

PROMPT_TEMPLATE = """Ты оцениваешь ответы модели по трём метрикам (0-1):

1. Faithfulness — соответствует ли ответ приведённому контексту (без выдумок)?
2. ContextPrecision — насколько ответ основан на нужных частях контекста?
3. AnswerRelevance — насколько ответ отвечает на сам вопрос?

Выведи JSON:
{{"faithfulness": x, "context_precision": y, "answer_relevance": z}}
---

Вопрос: {query}
Контекст: {retrieved}
Ответ: {answer}
"""

async def ollama_eval(data):
    async with httpx.AsyncClient() as client:
        results = []
        for i, item in enumerate(data, 1):
            prompt = PROMPT_TEMPLATE.format(**item)
            try:
                r = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": MODEL, "prompt": prompt, "stream": False},
                    timeout=120.0,
                )
                resp = r.json()["response"]
                metrics = json.loads(resp.split("{")[1].split("}")[0].join(["{","}"])) if "{" in resp else {}
            except Exception as e:
                metrics = {"faithfulness": 0, "context_precision": 0, "answer_relevance": 0}
            item.update(metrics)
            results.append(item)
            print(f"[{i}/{len(data)}] OK")
        return results

def make_html(results):
    avg = lambda k: sum(x.get(k,0) for x in results)/len(results)
    html = f"""
    <html><body><h2>Self-CRAG Evaluation</h2>
    <p>Samples: {len(results)}</p>
    <table border=1 cellpadding=6>
    <tr><th>Metric</th><th>Average</th></tr>
    <tr><td>Faithfulness</td><td>{avg("faithfulness"):.3f}</td></tr>
    <tr><td>ContextPrecision</td><td>{avg("context_precision"):.3f}</td></tr>
    <tr><td>AnswerRelevance</td><td>{avg("answer_relevance"):.3f}</td></tr>
    </table></body></html>"""
    open(OUT_HTML, "w", encoding="utf-8").write(html)
    print(f"[💾] HTML-отчёт сохранён в {OUT_HTML}")

async def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = [json.loads(x) for x in f]
    results = await ollama_eval(data)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    make_html(results)

if __name__ == "__main__":
    asyncio.run(main())
