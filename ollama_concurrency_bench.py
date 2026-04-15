#!/usr/bin/env python3
"""
Ollama Concurrency Benchmark — Qwen 3.5 Parallelism Investigation
==================================================================
Fires N simultaneous requests against the Ollama API and measures:
  - Individual response time (TTFT + generation)
  - True temporal overlap between requests
  - Aggregate throughput (tokens/s)
  - Detection of serialization vs. true parallelism
Usage:
    python ollama_concurrency_bench.py
    python ollama_concurrency_bench.py --model qwen3:32b --requests 4
    python ollama_concurrency_bench.py --model llama3.1:8b --requests 8
Requirements:
    pip install aiohttp
"""
from __future__ import annotations
import asyncio
import argparse
import json
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional
try:
    import aiohttp
except ImportError:
    print(
        "ERROR: 'aiohttp' is required but not installed.\n"
        "Install it with:  pip install aiohttp",
        file=sys.stderr,
    )
    sys.exit(1)
if sys.platform == "win32":
    stdout_reconfigured = False
    stderr_reconfigured = False
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        stdout_reconfigured = True
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        stderr_reconfigured = True
    if not (stdout_reconfigured and stderr_reconfigured):
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
DEFAULT_MODEL = "qwen3.5:35b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_NUM_REQUESTS = 10
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT = 300  
PROMPTS = [
    "Explain the concept of entropy in thermodynamics in 3 sentences.",
    "Write a Python function that implements binary search.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Explain how a neural network learns through backpropagation.",
    "What is the significance of the Turing Test?",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the CAP theorem in distributed systems.",
    "What are the SOLID principles in software engineering?",
    "Describe the difference between symmetric and asymmetric encryption.",
    "How does garbage collection work in Java?",
    "Explain the concept of eventual consistency.",
    "What is the difference between a process and a thread?",
    "Describe how DNS resolution works.",
    "Explain the MapReduce programming model.",
    "What are the benefits of microservices architecture?",
]
@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: int
    model: str
    prompt_preview: str
    start_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: float = 0.0
    tokens_generated: int = 0
    total_response: str = ""
    error: Optional[str] = None
    status: str = "pending"
    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token (seconds)."""
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None
    @property
    def total_time(self) -> float:
        """Total request time (seconds)."""
        return self.end_time - self.start_time
    @property
    def tokens_per_second(self) -> float:
        """Generation speed (tokens/s)."""
        gen_time = self.total_time - (self.ttft if self.ttft is not None else 0)
        if gen_time > 0 and self.tokens_generated > 0:
            return self.tokens_generated / gen_time
        return 0.0
@dataclass
class BenchmarkReport:
    """Consolidated benchmark report."""
    model: str
    num_requests: int
    max_tokens: int
    results: List[RequestResult] = field(default_factory=list)
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0
    @property
    def wall_clock_time(self) -> float:
        return self.wall_clock_end - self.wall_clock_start
    @property
    def successful_results(self) -> List[RequestResult]:
        return [r for r in self.results if r.status == "success"]
    @property
    def failed_results(self) -> List[RequestResult]:
        return [r for r in self.results if r.status == "error"]
async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    request_id: int,
    max_tokens: int,
    timeout: int,
) -> RequestResult:
    """Sends a streaming request to Ollama and collects metrics."""
    result = RequestResult(
        request_id=request_id,
        model=model,
        prompt_preview=prompt[:60] + "..." if len(prompt) > 60 else prompt,
    )
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        },
    }
    result.start_time = time.monotonic()
    response_chunks: List[str] = []
    decode_errors = 0
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.status = "error"
                result.end_time = time.monotonic()
                return result
            stream_done = False
            async for line in response.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    decode_errors += 1
                    continue
                if chunk.get("response") and result.first_token_time is None:
                    result.first_token_time = time.monotonic()
                if chunk.get("response"):
                    response_chunks.append(chunk["response"])
                if chunk.get("done", False):
                    result.tokens_generated = int(chunk.get("eval_count", 0))
                    stream_done = True
                    break
            if not stream_done and decode_errors > 0:
                result.error = (
                    f"Stream ended without done signal "
                    f"({decode_errors} JSON decode error(s))"
                )
                result.status = "error"
                result.end_time = time.monotonic()
                return result
        result.total_response = "".join(response_chunks)
        result.status = "success"
    except asyncio.TimeoutError:
        result.error = f"Timeout after {timeout}s"
        result.status = "error"
    except aiohttp.ClientError as e:
        result.error = f"Connection error: {str(e)}"
        result.status = "error"
    except Exception as e:
        result.error = f"Unexpected error: {str(e)}"
        result.status = "error"
    result.end_time = time.monotonic()
    return result
async def run_benchmark(
    model: str,
    num_requests: int,
    max_tokens: int,
    base_url: str,
    timeout: int,
) -> BenchmarkReport:
    """Fires N requests simultaneously and collects results."""
    report = BenchmarkReport(
        model=model,
        num_requests=num_requests,
        max_tokens=max_tokens,
    )
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    print(f"\n{'='*72}")
    print(f"  OLLAMA CONCURRENCY BENCHMARK")
    print(f"  Model:        {model}")
    print(f"  Requests:     {num_requests} (simultaneous)")
    print(f"  Max Tokens:   {max_tokens}")
    print(f"  Endpoint:     {base_url}")
    print(f"{'='*72}\n")
    print(f"[{time.strftime('%H:%M:%S')}] Firing {num_requests} simultaneous requests...")
    connector = aiohttp.TCPConnector(limit=num_requests + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        report.wall_clock_start = time.monotonic()
        tasks = [
            send_request(session, base_url, model, prompt, i, max_tokens, timeout)
            for i, prompt in enumerate(prompts)
        ]
        report.results = await asyncio.gather(*tasks)
        report.wall_clock_end = time.monotonic()
    return report
def analyze_concurrency(report: BenchmarkReport) -> dict:
    """Analyzes whether there was real parallelism or serialization."""
    successful = report.successful_results
    if len(successful) < 2:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "overlap_ratio": 0.0,
            "theoretical_speedup": 0.0,
        }
    intervals = [
        (r.start_time, r.end_time) for r in successful
    ]
    intervals.sort(key=lambda x: x[0])
    total_span = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    sum_individual = sum(r.total_time for r in successful)
    num_valid = len(successful)
    if total_span > 0:
        speedup = sum_individual / total_span
    else:
        speedup = 0.0
    if num_valid > 1:
        overlap_ratio = (speedup - 1.0) / (num_valid - 1.0)
        overlap_ratio = max(0.0, min(1.0, overlap_ratio))
    else:
        overlap_ratio = 0.0
    if overlap_ratio > 0.6:
        verdict = "PARALLEL"
    elif overlap_ratio > 0.15:
        verdict = "PARTIAL_PARALLEL"
    else:
        verdict = "SERIALIZED"
    return {
        "verdict": verdict,
        "overlap_ratio": overlap_ratio,
        "total_span": total_span,
        "sum_individual": sum_individual,
        "theoretical_speedup": speedup,
    }
def print_report(report: BenchmarkReport, output_dir: str = "bench_results"):
    """Prints the detailed benchmark report."""
    successful = report.successful_results
    failed = report.failed_results
    analysis = analyze_concurrency(report)
    print(f"\n{'='*72}")
    print(f"  RESULTS")
    print(f"{'='*72}\n")
    print(f"  {'ID':>3}  {'Status':>8}  {'TTFT(s)':>8}  {'Total(s)':>9}  {'Tokens':>6}  {'Tok/s':>7}  {'Start':>8}")
    print(f"  {'---':>3}  {'--------':>8}  {'-------':>8}  {'---------':>9}  {'------':>6}  {'-----':>7}  {'-----':>8}")
    base_time = report.wall_clock_start
    for r in sorted(report.results, key=lambda x: x.start_time):
        status = "[OK]" if r.status == "success" else "[FAIL]"
        ttft = f"{r.ttft:.3f}" if r.ttft is not None else "N/A"
        total = f"{r.total_time:.3f}"
        tokens = str(r.tokens_generated)
        tps = f"{r.tokens_per_second:.1f}" if r.tokens_per_second > 0 else "N/A"
        start_offset = f"+{r.start_time - base_time:.3f}"
        print(f"  {r.request_id:>3}  {status:>8}  {ttft:>8}  {total:>9}  {tokens:>6}  {tps:>7}  {start_offset:>8}")
        if r.error:
            print(f"       \\-- Error: {r.error[:70]}")
    print(f"\n{'-'*72}")
    print(f"  SUMMARY")
    print(f"{'-'*72}")
    print(f"  Successes:         {len(successful)}/{report.num_requests}")
    print(f"  Failures:          {len(failed)}/{report.num_requests}")
    print(f"  Wall Clock Total:  {report.wall_clock_time:.3f}s")
    if successful:
        ttft_values = [r.ttft for r in successful if r.ttft is not None]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None
        avg_tps = sum(r.tokens_per_second for r in successful) / len(successful)
        total_tokens = sum(r.tokens_generated for r in successful)
        aggregate_tps = total_tokens / report.wall_clock_time if report.wall_clock_time > 0 else 0
        avg_ttft_display = f"{avg_ttft:.3f}s" if avg_ttft is not None else "N/A"
        print(f"  Avg TTFT:          {avg_ttft_display}")
        print(f"  Avg Tok/s (indiv): {avg_tps:.1f}")
        print(f"  Total Tokens:      {total_tokens}")
        print(f"  Aggregate Tok/s:   {aggregate_tps:.1f}")
    print(f"\n{'-'*72}")
    print(f"  CONCURRENCY ANALYSIS")
    print(f"{'-'*72}")
    verdict_label = {
        "PARALLEL": "[+] PARALLEL",
        "PARTIAL_PARALLEL": "[~] PARTIAL PARALLELISM",
        "SERIALIZED": "[!] SERIALIZED",
        "INSUFFICIENT_DATA": "[?] INSUFFICIENT DATA",
    }
    print(f"  Verdict:           {verdict_label.get(analysis['verdict'], analysis['verdict'])}")
    print(f"  Overlap Ratio:     {analysis.get('overlap_ratio', 0):.2%}")
    print(f"  Theoretical Speedup: {analysis.get('theoretical_speedup', 0):.2f}x")
    if analysis["verdict"] == "SERIALIZED":
        print(f"\n  [WARNING] SERIALIZATION DETECTED!")
        print(f"  Requests are being processed one at a time.")
        print(f"  Sum of individual times ~= Wall Clock Total.")
        print(f"  Check: OLLAMA_NUM_PARALLEL, model architecture (GDN/hybrid).")
    elif analysis["verdict"] == "PARALLEL":
        print(f"\n  [OK] REAL PARALLELISM DETECTED!")
        print(f"  Requests are being processed concurrently.")
    elif analysis["verdict"] == "PARTIAL_PARALLEL":
        print(f"\n  [INFO] PARTIAL PARALLELISM!")
        print(f"  Some overlap detected, but not ideal. May be slot or resource limitation.")
    print(f"\n{'-'*72}")
    print(f"  TIMELINE (normalized scale)")
    print(f"{'-'*72}")
    if successful:
        min_start = min(r.start_time for r in successful)
        max_end = max(r.end_time for r in successful)
        span = max_end - min_start
        if span > 0:
            bar_width = 50
            for r in sorted(successful, key=lambda x: x.request_id):
                offset = int(((r.start_time - min_start) / span) * bar_width)
                length = max(1, int(((r.end_time - r.start_time) / span) * bar_width))
                bar = " " * offset + "#" * length
                print(f"  Req {r.request_id:>2}: |{bar:<{bar_width}}| {r.total_time:.1f}s")
    print(f"\n{'='*72}\n")
    export = {
        "model": report.model,
        "num_requests": report.num_requests,
        "max_tokens": report.max_tokens,
        "wall_clock_time": report.wall_clock_time,
        "analysis": analysis,
        "results": [
            {
                "id": r.request_id,
                "status": r.status,
                "ttft": r.ttft,
                "total_time": r.total_time,
                "tokens": r.tokens_generated,
                "tps": r.tokens_per_second,
                "start_offset": r.start_time - report.wall_clock_start,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    safe_model = "".join(c if (c.isalnum() or c in "._-") else "_" for c in report.model)
    safe_model = safe_model.lstrip("._-") or "model"
    resolved_dir = os.path.realpath(output_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    filename = f"bench_result_{safe_model}_{int(time.time())}.json"
    export_path = os.path.realpath(os.path.join(resolved_dir, filename))
    if os.path.commonpath([resolved_dir, export_path]) != resolved_dir:
        raise ValueError("Resolved export path escapes output directory")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  [FILE] Results exported to: {export_path}\n")
def main():
    parser = argparse.ArgumentParser(
        description="Ollama Concurrency Benchmark — Qwen 3.5 Investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ollama_concurrency_bench.py
  python ollama_concurrency_bench.py --model qwen3:32b --requests 4
  python ollama_concurrency_bench.py --model llama3.1:8b --requests 8 --base-url http://192.168.1.100:11434
        """,
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f"Number of simultaneous requests (default: {DEFAULT_NUM_REQUESTS})",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--base-url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"Ollama Base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per request in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="bench_results",
        help="Directory for JSON result exports (default: bench_results)",
    )
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    report = asyncio.run(
        run_benchmark(
            model=args.model,
            num_requests=args.requests,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    )
    print_report(report, output_dir=args.output_dir)
    analysis = analyze_concurrency(report)
    if analysis["verdict"] == "SERIALIZED":
        sys.exit(1)  
    elif analysis["verdict"] == "PARALLEL":
        sys.exit(0)  
    else:
        sys.exit(2)  
if __name__ == "__main__":
    main()
