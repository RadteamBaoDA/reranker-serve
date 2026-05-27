/* Reranker Admin — performance trend charts (uPlot).
 * Seeds each chart from /admin/api/metrics/history?window=, then polls
 * incrementally with ?since= and appends, honouring window + pause controls. */
(function () {
  "use strict";
  if (typeof uPlot === "undefined") {
    document.getElementById("chart-status").textContent = "uPlot failed to load";
    return;
  }

  var WINDOW = 1800;      // seconds shown
  var POLL_MS = 5000;     // incremental poll cadence
  var PAUSE = false;
  var SAMPLES = [];       // rolling array of sample objects
  var lastT = 0;          // newest sample timestamp seen
  var serverNow = Date.now() / 1000;
  var charts = {};
  var timer = null;

  var AXIS = "#8b94a7", GRID = "rgba(255,255,255,.06)";

  function baseOpts(title, height) {
    return {
      title: title,
      width: 400,
      height: height || 210,
      cursor: { y: false },
      legend: { live: true },
      axes: [
        { stroke: AXIS, grid: { stroke: GRID }, ticks: { stroke: GRID } },
        { stroke: AXIS, grid: { stroke: GRID }, ticks: { stroke: GRID } },
      ],
    };
  }

  function makeChart(elId, title, seriesDefs, scales, axes) {
    var el = document.getElementById(elId);
    if (!el) return null;
    var opts = baseOpts(title);
    opts.width = el.clientWidth || 420;
    opts.series = [{}].concat(seriesDefs);
    if (scales) opts.scales = scales;
    if (axes) opts.axes = axes;
    var data = [[]];
    for (var i = 0; i < seriesDefs.length; i++) data.push([]);
    return new uPlot(opts, data, el);
  }

  function rightAxis(label) {
    return { side: 1, scale: label, stroke: AXIS, grid: { show: false }, ticks: { stroke: GRID } };
  }

  function init() {
    charts.latency = makeChart("chart-latency", "Latency (ms)", [
      { label: "p50", stroke: "#3b82f6", width: 2 },
      { label: "p95", stroke: "#ec4899", width: 2 },
    ]);
    charts.throughput = makeChart("chart-throughput", "Throughput", [
      { label: "req/s", stroke: "#22c55e", width: 2 },
      { label: "pairs/s", stroke: "#f59e0b", width: 2, scale: "pairs" },
    ],
      { x: {}, y: {}, pairs: {} },
      [
        { stroke: AXIS, grid: { stroke: GRID } },
        { stroke: AXIS, grid: { stroke: GRID } },
        rightAxis("pairs"),
      ]
    );
    charts.gpu = makeChart("chart-gpu", "GPU util & VRAM (%)", [
      { label: "util %", stroke: "#8b5cf6", width: 2 },
      { label: "vram %", stroke: "#06b6d4", width: 2 },
    ], { x: {}, y: { range: [0, 100] } });
    charts.queue = makeChart("chart-queue", "Queue depth & occupancy", [
      { label: "running", stroke: "#38bdf8", width: 2 },
      { label: "waiting", stroke: "#ef4444", width: 2 },
      { label: "occ %", stroke: "#a3a3a3", width: 1, dash: [4, 3], scale: "pct" },
    ],
      { x: {}, y: {}, pct: { range: [0, 100] } },
      [
        { stroke: AXIS, grid: { stroke: GRID } },
        { stroke: AXIS, grid: { stroke: GRID } },
        rightAxis("pct"),
      ]
    );
  }

  function col(key) { return SAMPLES.map(function (s) { return s[key]; }); }

  function redraw() {
    var xs = SAMPLES.map(function (s) { return s.t; });
    if (charts.latency) charts.latency.setData([xs, col("p50_ms"), col("p95_ms")]);
    if (charts.throughput) charts.throughput.setData([xs, col("rps"), col("pairs_s")]);
    if (charts.gpu) charts.gpu.setData([xs, col("gpu_util"), col("gpu_mem_pct")]);
    if (charts.queue) charts.queue.setData([xs, col("running"), col("waiting"), col("batch_occupancy_pct")]);
  }

  function trim() {
    var cutoff = serverNow - WINDOW;
    SAMPLES = SAMPLES.filter(function (s) { return s.t >= cutoff; });
  }

  function status(msg) { document.getElementById("chart-status").textContent = msg; }

  function seed() {
    fetch("/admin/api/metrics/history?window=" + WINDOW, { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        SAMPLES = d.samples || [];
        serverNow = d.server_now || Date.now() / 1000;
        lastT = SAMPLES.length ? SAMPLES[SAMPLES.length - 1].t : 0;
        redraw();
        status(SAMPLES.length + " pts · " + (WINDOW / 60) + "m");
      })
      .catch(function () { status("history unavailable"); });
  }

  function poll() {
    if (PAUSE) return;
    fetch("/admin/api/metrics/history?since=" + lastT, { credentials: "same-origin" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        serverNow = d.server_now || serverNow;
        var fresh = d.samples || [];
        if (fresh.length) {
          SAMPLES = SAMPLES.concat(fresh);
          lastT = SAMPLES[SAMPLES.length - 1].t;
        }
        trim();
        redraw();
        status(SAMPLES.length + " pts · " + (WINDOW / 60) + "m");
      })
      .catch(function () { /* transient */ });
  }

  function resize() {
    Object.keys(charts).forEach(function (k) {
      var u = charts[k];
      if (u) u.setSize({ width: u.root.parentNode.clientWidth || 420, height: 210 });
    });
  }

  // Controls
  document.getElementById("window-seg").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) return;
    WINDOW = parseInt(b.dataset.window, 10);
    [].forEach.call(this.children, function (c) { c.classList.remove("active"); });
    b.classList.add("active");
    seed();
  });
  document.getElementById("pause-btn").addEventListener("click", function () {
    PAUSE = !PAUSE;
    this.classList.toggle("paused", PAUSE);
    this.textContent = PAUSE ? "▶ Resume" : "⏸ Pause";
  });
  var rt;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 150); });

  init();
  seed();
  timer = setInterval(poll, POLL_MS);
})();
