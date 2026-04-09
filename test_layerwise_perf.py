import asyncio
import aiohttp
import time
import argparse
import numpy as np
from tabulate import tabulate # 请先 pip install tabulate

def generate_long_prompt(length):
    # 粗略估计：1个token约等于4个字符，这里按字符构造
    base_text = "The quick brown fox jumps over the lazy dog. "
    return (base_text * (length // 10 + 1))[:length]

async def measure_request(session, url, model_name, prompt_len):
    prompt = generate_long_prompt(prompt_len)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 32, # 生成少量内容以减小 Decode 阶段干扰
        "temperature": 0.0,
        "stream": True
    }
    
    start_time = time.perf_counter()
    ttft = None
    last_token_time = None
    tokens_received = 0
    
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    content = line.decode('utf-8')
                    if "[DONE]" in content:
                        break
                    
                    tokens_received += 1
                    current_time = time.perf_counter()
                    
                    if ttft is None:
                        ttft = current_time - start_time
                    last_token_time = current_time
        
        e2e_latency = last_token_time - start_time
        # TPOT = (总时间 - 首字时间) / (总token数 - 1)
        tpot = (e2e_latency - ttft) / (tokens_received - 1) if tokens_received > 1 else 0
        
        return {
            "ttft": ttft,
            "tpot": tpot,
            "e2e": e2e_latency,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def run_batch(url, model, length, n_requests, concurrency):
    print(f"正在测试长度: {length} ...", end="\r")
    async with aiohttp.ClientSession() as session:
        tasks = [measure_request(session, url, model, length) for _ in range(n_requests)]
        results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r["success"]]
    if not valid_results:
        return None
    
    return {
        "Length": f"{length//1024}k",
        "Avg TTFT (s)": np.mean([r["ttft"] for r in valid_results]),
        "P99 TTFT (s)": np.percentile([r["ttft"] for r in valid_results], 99),
        "Avg TPOT (ms)": np.mean([r["tpot"] for r in valid_results]) * 1000,
        "Avg E2E (s)": np.mean([r["e2e"] for r in valid_results])
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # 测试配置
    test_lengths = [1024, 4096, 16384, 32768]
    n_requests = 10
    concurrency = 1 # 建议设为1以观察纯粹的传输开销，不被调度干扰

    all_stats = []
    print(f"开始 Mooncake Layer-wise 性能评估")
    print(f"目标 URL: {args.url} | 模型: {args.model}")
    print("-" * 50)

    for length in test_lengths:
        stat = await run_batch(args.url, args.model, length, n_requests, concurrency)
        if stat:
            all_stats.append(stat)

    print("\n" + tabulate(all_stats, headers="keys", tablefmt="grid", floatfmt=".4f"))

if __name__ == "__main__":
    asyncio.run(main())