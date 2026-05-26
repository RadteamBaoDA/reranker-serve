#!/usr/bin/env python3
"""
Live TUI dashboard for benchmarking the reranker service.

Renders five panels that update ~10x/sec while the benchmark runs:
  - Header: target URL, model, device, elapsed time, live RPS
  - Progress bar: completed / total
  - Client latency: avg / p50 / p95 / p99 / max in ms
  - Status code breakdown: 200, 503 (backpressure), 0 (transport error), etc.
  - Server engine /stats: batch occupancy, inflight batches, queue wait,
    inference latency, throughput pairs/sec (polled every --stats-interval)
  - Recent requests: scrolling list of the last 12 completions

Dependency: rich (pip install rich). The plain stdout benchmark
benchmark_concurrent.py has no rich dependency and is the right choice if
you don't have rich installed or need machine-parseable output.

Usage:
    python benchmark_tui.py --url http://localhost:8000 \
        --requests 1000 --concurrency 32 --documents 10
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

try:
    import httpx
except ImportError:
    raise SystemExit("benchmark_tui.py requires httpx (pip install httpx)")

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    raise SystemExit(
        "benchmark_tui.py requires rich (pip install rich). "
        "Use benchmark_concurrent.py for a no-dependency stdout benchmark."
    )


BASE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to learn representations.",
    "The weather is sunny today with clear skies.",
    "Natural language processing helps computers understand human language.",
    "Python is a popular programming language for data science.",
    "Transformers have revolutionized NLP with attention mechanisms.",
    "Random forests are ensemble learning methods for classification.",
    "Gradient descent is an optimization algorithm used in machine learning.",
    "Convolutional neural networks are commonly used for image recognition.",
    "Reinforcement learning trains agents through reward-based feedback.",
]


# ---------- shared state ----------

@dataclass
class State:
    total: int = 0
    completed: int = 0
    in_flight: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    status_counts: Dict[int, int] = field(default_factory=dict)
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=12))
    server_stats: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def rps(self) -> float:
        return self.completed / self.elapsed if self.elapsed > 0 else 0.0


# ---------- workers ----------

async def send_one(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    documents: List[str],
    request_id: int,
    state: State,
) -> None:
    state.in_flight += 1
    start = time.perf_counter()
    code = 0
    top_score: Optional[float] = None
    retry_after: Optional[str] = None
    error_text: Optional[str] = None

    try:
        resp = await client.post(
            f"{url}/rerank",
            json={"query": query, "documents": documents, "top_n": 3},
            timeout=60.0,
        )
        code = resp.status_code
        if code == 200:
            try:
                body = resp.json()
                results = body.get("results") or []
                if results:
                    top_score = results[0].get("relevance_score")
            except Exception:
                pass
        else:
            retry_after = resp.headers.get("Retry-After")
            error_text = (resp.text or "")[:120]
    except Exception as e:
        error_text = str(e)[:120]
    finally:
        state.in_flight -= 1
        latency_ms = (time.perf_counter() - start) * 1000.0
        state.completed += 1
        state.latencies_ms.append(latency_ms)
        state.status_counts[code] = state.status_counts.get(code, 0) + 1
        if error_text:
            state.last_error = f"[#{request_id} status={code}] {error_text}"

        extra = ""
        if top_score is not None:
            extra = f" top={top_score:.3f}"
        elif retry_after is not None:
            extra = f" retry_after={retry_after}"
        elif error_text:
            extra = f" err={error_text[:40]}"
        state.recent.appendleft(
            f"[t={state.elapsed:6.2f}s] #{request_id:04d} {code} {latency_ms:7.1f}ms{extra}"
        )


async def stats_poller(
    url: str,
    interval: float,
    state: State,
    stop: asyncio.Event,
) -> None:
    if interval <= 0:
        return
    async with httpx.AsyncClient() as client:
        while not stop.is_set():
            try:
                r = await client.get(f"{url}/stats", timeout=2.0)
                if r.status_code == 200:
                    payload = r.json()
                    state.server_stats = payload.get("stats", payload)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


# ---------- rendering ----------

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="progress", size=3),
        Layout(name="middle", size=9),
        Layout(name="engine", size=6),
        Layout(name="recent", ratio=1),
    )
    layout["middle"].split_row(
        Layout(name="latency"),
        Layout(name="status"),
    )
    return layout


def _fmt(v: Any, spec: str = "{:.2f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return spec.format(v)
    return str(v)


def render_header(state: State, args: argparse.Namespace) -> Panel:
    h = state.health
    body = (
        f"URL: {args.url}    elapsed: {state.elapsed:6.2f}s    "
        f"concurrency: {args.concurrency}    RPS: {state.rps:6.2f}\n"
        f"Model: {h.get('model', '?')}    "
        f"Device: {h.get('device', '?')}    "
        f"Engine: {h.get('engine_mode', '?')}"
    )
    return Panel(body, title="[bold cyan]Reranker Benchmark", border_style="cyan")


def render_progress(state: State) -> Panel:
    pct = (state.completed / state.total * 100.0) if state.total else 0.0
    bar_width = 50
    filled = int(pct / 100.0 * bar_width)
    bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_width - filled) + "[/dim]"
    in_flight = state.in_flight
    body = f"{bar}  {state.completed}/{state.total} ({pct:5.1f}%)  in-flight: {in_flight}"
    return Panel(body, title="[bold green]Progress", border_style="green")


def render_latency(state: State) -> Panel:
    lats = list(state.latencies_ms)
    if not lats:
        return Panel("(no completed requests yet)", title="[bold yellow]Client Latency",
                     border_style="yellow")
    sorted_lat = sorted(lats)
    n = len(sorted_lat)
    avg = statistics.mean(lats)
    p50 = sorted_lat[n // 2]
    p95 = sorted_lat[max(0, int(n * 0.95) - 1)]
    p99 = sorted_lat[max(0, int(n * 0.99) - 1)]
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="bold")
    table.add_column(justify="right")
    table.add_row("Samples", str(n))
    table.add_row("Avg", f"{avg:7.2f} ms")
    table.add_row("p50", f"{p50:7.2f} ms")
    table.add_row("p95", f"{p95:7.2f} ms")
    table.add_row("p99", f"{p99:7.2f} ms")
    table.add_row("Max", f"{max(lats):7.2f} ms")
    return Panel(table, title="[bold yellow]Client Latency",
                 border_style="yellow")


def render_status(state: State) -> Panel:
    if not state.status_counts:
        return Panel("(no requests yet)", title="[bold magenta]Status Codes",
                     border_style="magenta")
    total = sum(state.status_counts.values())
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="bold")
    table.add_column(justify="right")
    table.add_column(justify="right")
    for code in sorted(state.status_counts):
        n = state.status_counts[code]
        pct = n / total * 100
        if code == 0:
            label = "[red]transport_err[/red]"
        elif code == 200:
            label = "[green]200[/green]"
        elif code in (503, 504):
            label = f"[yellow]{code}[/yellow]"
        else:
            label = f"[red]{code}[/red]"
        table.add_row(label, str(n), f"{pct:5.1f}%")
    return Panel(table, title="[bold magenta]Status Codes",
                 border_style="magenta")


def render_engine(state: State) -> Panel:
    s = state.server_stats or {}
    if not s:
        return Panel("(no /stats yet — is the server running?)",
                     title="[bold blue]Server Engine /stats", border_style="blue")
    body = (
        f"Total batches: {_fmt(s.get('total_batches'), '{}')}    "
        f"Avg batch: {_fmt(s.get('avg_batch_size'))}    "
        f"Occupancy: {_fmt(s.get('batch_occupancy_pct'))}%\n"
        f"Inflight: {_fmt(s.get('inflight_batches'), '{}')}    "
        f"Pending: {_fmt(s.get('pending_requests'), '{}')}    "
        f"Throughput: {_fmt(s.get('throughput_pairs_per_sec'))} pairs/sec\n"
        f"Queue wait p95: {_fmt(s.get('queue_wait_p95_ms'))} ms    "
        f"Inference p95: {_fmt(s.get('inference_latency_p95_ms'))} ms"
    )
    return Panel(body, title="[bold blue]Server Engine /stats",
                 border_style="blue")


def render_recent(state: State) -> Panel:
    if not state.recent:
        body = "(no requests yet)"
    else:
        body = "\n".join(state.recent)
    title = f"[bold]Recent Requests (last {len(state.recent)})"
    return Panel(body, title=title, border_style="white")


def refresh(layout: Layout, state: State, args: argparse.Namespace) -> None:
    layout["header"].update(render_header(state, args))
    layout["progress"].update(render_progress(state))
    layout["latency"].update(render_latency(state))
    layout["status"].update(render_status(state))
    layout["engine"].update(render_engine(state))
    layout["recent"].update(render_recent(state))


async def renderer(
    layout: Layout, state: State, args: argparse.Namespace, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        refresh(layout, state, args)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
    refresh(layout, state, args)  # final tick


# ---------- driver ----------

async def run(args: argparse.Namespace) -> None:
    state = State(total=args.requests)
    documents = BASE_DOCUMENTS[:args.documents]
    query = args.query

    # Initial health check (best effort — we still render the dashboard
    # so the user can see what's wrong even if the server is down).
    async with httpx.AsyncClient() as health_client:
        try:
            h = await health_client.get(f"{args.url}/health", timeout=5.0)
            if h.status_code == 200:
                state.health = h.json()
        except Exception:
            pass

    layout = make_layout()
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(client, i):
        async with semaphore:
            await send_one(client, args.url, query, documents, i, state)

    console = Console()
    with Live(layout, refresh_per_second=10, console=console, screen=False, redirect_stderr=False):
        renderer_task = asyncio.create_task(renderer(layout, state, args, stop))
        poller_task = asyncio.create_task(stats_poller(args.url, args.stats_interval, state, stop))

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(bounded(client, i) for i in range(args.requests)))

        # Give the poller one last tick after the burst, then stop.
        if args.stats_interval > 0:
            await asyncio.sleep(min(args.stats_interval, 1.0))
        stop.set()
        await asyncio.gather(renderer_task, poller_task, return_exceptions=True)

    # Print final summary outside the Live region.
    console.print()
    console.rule("[bold]Final Summary")
    lats = list(state.latencies_ms)
    console.print(f"Total time: {state.elapsed:.2f}s  |  RPS: {state.rps:.2f}")
    console.print(f"Status codes: {dict(sorted(state.status_counts.items()))}")
    if lats:
        sorted_lat = sorted(lats)
        console.print(
            f"Latency ms — avg={statistics.mean(lats):.2f} "
            f"p50={sorted_lat[len(sorted_lat) // 2]:.2f} "
            f"p95={sorted_lat[max(0, int(len(sorted_lat) * 0.95) - 1)]:.2f} "
            f"p99={sorted_lat[max(0, int(len(sorted_lat) * 0.99) - 1)]:.2f} "
            f"max={max(lats):.2f}"
        )
    if state.last_error:
        console.print(f"[red]Last error:[/red] {state.last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reranker benchmark with live TUI dashboard"
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the reranker service")
    parser.add_argument("--requests", type=int, default=100,
                        help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Maximum concurrent in-flight requests")
    parser.add_argument("--documents", type=int, default=5,
                        help="Number of documents per request")
    parser.add_argument("--query", default="What is machine learning?",
                        help="The query string to send")
    parser.add_argument("--stats-interval", type=float, default=1.0,
                        help="Poll /stats every N seconds (0 to disable)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
