// plt_technicals.js
// Nifty candlestick + Bollinger Bands + RSI + Awesome Oscillator
// Assumes:
//  - LightweightCharts v5 standalone is loaded in <head>
//  - This script is loaded after the DOM (e.g. end of <body>)
//  - data/nifty_technicals.json exists and is valid JSON

(function () {
  const charts = [];

  async function loadData() {
    try {
      const res = await fetch("data/nifty_technicals.json");
      if (!res.ok) {
        console.error("Failed to load data/nifty_technicals.json, status:", res.status);
        return [];
      }
      const data = await res.json();
      if (!Array.isArray(data)) {
        console.error("Unexpected JSON shape for nifty_technicals.json");
        return [];
      }
      return data;
    } catch (err) {
      console.error("Error loading nifty_technicals.json:", err);
      return [];
    }
  }

  function createMainChart(data) {
    const container = document.getElementById("nifty-candles");
    if (!container) {
      console.warn("No #nifty-candles container found");
      return null;
    }

    const chartApi = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: "#151a30" },
        textColor: "#d6e4ff",
      },
      grid: {
        vertLines: { color: "rgba(200, 200, 255, 0.05)" },
        horzLines: { color: "rgba(200, 200, 255, 0.05)" },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
      },
    });

    const candleSeries = chartApi.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderDownColor: "#ef5350",
      borderUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      wickUpColor: "#26a69a",
    });

    const bbMidSeries = chartApi.addSeries(LightweightCharts.LineSeries, {
      color: "rgba(255, 255, 255, 0.7)",
      lineWidth: 1,
    });
    const bbUpperSeries = chartApi.addSeries(LightweightCharts.LineSeries, {
      color: "rgba(129, 212, 250, 0.8)",
      lineWidth: 1,
    });
    const bbLowerSeries = chartApi.addSeries(LightweightCharts.LineSeries, {
      color: "rgba(129, 212, 250, 0.8)",
      lineWidth: 1,
    });

    const candleData = data.map(d => ({
      time: d.time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    const bbMidData = data.map(d => ({
      time: d.time,
      value: d.bb_mid,
    }));
    const bbUpperData = data.map(d => ({
      time: d.time,
      value: d.bb_upper,
    }));
    const bbLowerData = data.map(d => ({
      time: d.time,
      value: d.bb_lower,
    }));

    candleSeries.setData(candleData);
    bbMidSeries.setData(bbMidData);
    bbUpperSeries.setData(bbUpperData);
    bbLowerSeries.setData(bbLowerData);

    charts.push(chartApi);
    return chartApi;
  }

  function createRsiChart(data) {
    const container = document.getElementById("nifty-rsi");
    if (!container) {
      console.warn("No #nifty-rsi container found");
      return null;
    }

    const chartApi = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: "#151a30" },
        textColor: "#d6e4ff",
      },
      grid: {
        vertLines: { color: "rgba(200, 200, 255, 0.05)" },
        horzLines: { color: "rgba(200, 200, 255, 0.05)" },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const rsiSeries = chartApi.addSeries(LightweightCharts.HistogramSeries, {
      priceFormat: { type: "price", precision: 2, minMove: 0.1 },
    });

    const rsiData = data.map(d => {
      const v = d.rsi;
      let color = "rgba(144, 202, 249, 0.7)"; // neutral
      if (v >= 70) color = "rgba(239, 83, 80, 0.85)";      // overbought
      else if (v <= 30) color = "rgba(102, 187, 106, 0.85)"; // oversold
      return {
        time: d.time,
        value: v,
        color,
      };
    });

    rsiSeries.setData(rsiData);

    rsiSeries.createPriceLine({
      price: 30,
      color: "rgba(102, 187, 106, 0.7)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: "30",
    });
    rsiSeries.createPriceLine({
      price: 70,
      color: "rgba(239, 83, 80, 0.7)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: "70",
    });

    charts.push(chartApi);
    return chartApi;
  }

  function createAoChart(data) {
    const container = document.getElementById("nifty-ao");
    if (!container) {
      console.warn("No #nifty-ao container found");
      return null;
    }

    const chartApi = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: "#151a30" },
        textColor: "#d6e4ff",
      },
      grid: {
        vertLines: { color: "rgba(200, 200, 255, 0.05)" },
        horzLines: { color: "rgba(200, 200, 255, 0.05)" },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const aoSeries = chartApi.addSeries(LightweightCharts.HistogramSeries, {
      priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
    });

    const aoData = data.map(d => ({
      time: d.time,
      value: d.ao,
      color: d.ao >= 0 ? "rgba(102, 187, 106, 0.8)" : "rgba(239, 83, 80, 0.8)",
    }));

    aoSeries.setData(aoData);

    aoSeries.createPriceLine({
      price: 0,
      color: "rgba(255, 255, 255, 0.5)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: "0",
    });

    charts.push(chartApi);
    return chartApi;
  }

  function syncTimeScales(mainChartApi, otherCharts) {
    if (!mainChartApi) return;
    const mainTimeScale = mainChartApi.timeScale();
    mainTimeScale.subscribeVisibleLogicalRangeChange(range => {
      if (!range) return;
      otherCharts.forEach(ch => {
        if (!ch) return;
        const ts = ch.timeScale();
        ts.setVisibleLogicalRange(range);
      });
    });
  }

  function updateLastUpdated(data) {
    if (!data || !data.length) return;
    const last = data[data.length - 1];
    const dt = new Date(last.time * 1000);
    const formatted =
      dt.getFullYear() +
      "-" +
      String(dt.getMonth() + 1).padStart(2, "0") +
      "-" +
      String(dt.getDate()).padStart(2, "0") +
      " " +
      String(dt.getHours()).padStart(2, "0") +
      ":" +
      String(dt.getMinutes()).padStart(2, "0");
    const el = document.getElementById("lastUpdated");
    if (el) {
      el.textContent = "Last data point: " + formatted + " (IST, delayed)";
    }
  }

  function resizeAllCharts() {
    const mainContainer = document.getElementById("nifty-candles");
    const rsiContainer = document.getElementById("nifty-rsi");
    const aoContainer = document.getElementById("nifty-ao");

    const sizes = [
      { chartApi: charts[0], container: mainContainer },
      { chartApi: charts[1], container: rsiContainer },
      { chartApi: charts[2], container: aoContainer },
    ];

    sizes.forEach(item => {
      if (!item.chartApi || !item.container) return;
      item.chartApi.applyOptions({
        width: item.container.clientWidth,
        height: item.container.clientHeight,
      });
    });
  }

  window.addEventListener("load", async () => {
    const data = await loadData();
    if (!data || !data.length) {
      console.error("No data available for nifty_technicals.json");
      return;
    }

    const mainChartApi = createMainChart(data);
    const rsiChartApi = createRsiChart(data);
    const aoChartApi = createAoChart(data);

    syncTimeScales(mainChartApi, [rsiChartApi, aoChartApi]);
    updateLastUpdated(data);

    window.addEventListener("resize", resizeAllCharts);
    resizeAllCharts();
  });
})();