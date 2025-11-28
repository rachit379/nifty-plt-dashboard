    const SNAPSHOT_URL = "data/nifty_snapshot.json";
    const EVENTS_URL = "data/events.csv";
    const NEWS_URL = "data/news.json";

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value ?? "—";
    }

    function formatChangePct(curr, prev) {
      if (curr == null || prev == null || prev === 0) return null;
      const pct = ((curr - prev) / prev) * 100;
      return pct;
    }

    function categoryClass(cat) {
      if (!cat) return "neutral";
      const c = cat.toLowerCase();
      if (c.includes("critical")) return "critical";
      if (c.includes("elevated")) return "elevated";
      if (c.includes("moderate")) return "moderate";
      if (c.includes("low")) return "low";
      if (c.includes("neutral")) return "neutral";
      return "neutral";
    }

    let globalSubmetricDefs = {};

    function updateSubmetricPills(component) {
      const pillsContainer = document.getElementById("regime-submetrics");
      const defsContainer = document.getElementById("regime-sub-defs");
      if (!pillsContainer || !defsContainer) return;

      pillsContainer.innerHTML = "";
      defsContainer.innerHTML = "";

      const list = globalSubmetricDefs[component] || [];
      if (!list.length) return;

      list.forEach((item) => {
        const obj = item.obj || {};
        const cls = categoryClass(obj.category);

        const pill = document.createElement("div");
        pill.className = "regime-sub-pill " + cls;
        pill.innerHTML = `
          <span class="regime-sub-label">${item.label}</span>
          <span class="regime-sub-value">${obj.value != null ? obj.value.toFixed(2) : "—"}</span>
          <span class="regime-sub-cat">${obj.category || "—"}</span>
        `;
        pillsContainer.appendChild(pill);

        const def = document.createElement("div");
        def.className = "regime-sub-defs-item";
        def.innerHTML = `
          <span class="regime-sub-defs-label">${item.label}:</span>
          <span>${obj.explanation || "No explanation available."}</span>
        `;
        defsContainer.appendChild(def);
      });
    }

    // Apply snapshot JSON
    function applySnapshot(data) {
      const regime = data.regime_state || {};
      const crit = regime.criticality || {};
      const shocks = regime.shocks || {};
      const contagion = regime.contagion || {};
      const direction = regime.direction || {};
      const optionsState = data.options_state || {};
      const spot = optionsState.spot_levels || {};
      const longPlan = optionsState.long_plan || {};
      const shortPlan = optionsState.short_plan || {};
      const regimeMeta = regime.regime || {};

      const critCats = crit.categories || {};
      const shocksCats = shocks.categories || {};
      const contagionCats = contagion.categories || {};
      const directionCats = (direction && direction.categories) || {};

      // Map of all submetrics per arm (for pills + definitions)
      globalSubmetricDefs = {
        C: [
          { label: "Dryness", obj: critCats.dryness_score || {} },
          { label: "IV vs RV", obj: critCats.ivrv_score || {} },
          { label: "Crowding", obj: critCats.crowding_score || {} },
          { label: "PCR", obj: critCats.pcr_score || {} },
        ],
        S: [
          { label: "Gap score", obj: shocksCats.gap_score || {} },
          { label: "Intraday score", obj: shocksCats.intraday_score || {} },
          { label: "Volume score", obj: shocksCats.volume_score || {} },
        ],
        K: [
          { label: "Breadth sync", obj: contagionCats.breadth_sync_score || {} },
          { label: "Wide move", obj: contagionCats.wide_move_score || {} },
          { label: "Index corr", obj: contagionCats.index_corr_score || {} },
        ],
        D: [
          { label: "Far OI bias", obj: directionCats.far_oi_bias || {} },
          { label: "Near OI bias", obj: directionCats.near_oi_bias || {} },
          { label: "PCR bias", obj: directionCats.pcr_bias || {} },
        ],
      };

      // Top hero
      setText("regime-name", regimeMeta.name || "—");
      setText("regime-description", regimeMeta.description || "No regime description available.");
      const asOf = regime.as_of || {};
      setText(
        "regime-asof",
        `Critical: ${asOf.critical || "—"} · Shocks: ${asOf.shock || "—"} · Contagion: ${asOf.contagion || "—"} · Direction: ${asOf.direction || "—"}`
      );

      // Nifty card
      const close = spot.close;
      const prevClose = spot.prev_close;
      const high = spot.prev_high;
      const low = spot.prev_low;
      setText("spot-asof", `Spot levels as of ${optionsState.as_of?.spot_levels || "—"}`);
      setText("nifty-spot", close != null ? close.toFixed(2) : "—");
      const changePct = formatChangePct(close, prevClose);
      const changeEl = document.getElementById("nifty-change");
      if (changeEl && changePct != null) {
        changeEl.textContent = `${(close - prevClose).toFixed(1)} (${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%)`;
        changeEl.classList.toggle("positive", changePct > 0);
        changeEl.classList.toggle("negative", changePct < 0);
      }
      setText("nifty-range", `Day range ${low != null ? low.toFixed(1) : "—"} – ${high != null ? high.toFixed(1) : "—"}`);
      const range20 = `${spot.low_20d?.toFixed(0) ?? "—"} – ${spot.high_20d?.toFixed(0) ?? "—"}`;
      const range52 = `${spot.low_52w?.toFixed(0) ?? "—"} – ${spot.high_52w?.toFixed(0) ?? "—"}`;
      setText("range-20d", range20);
      setText("range-52w", range52);
      setText("atr-14", spot.atr_14 != null ? spot.atr_14.toFixed(1) : "—");

      // Categories for each main arm
      const S_main = shocksCats.S || {};
      const K_main = contagionCats.K || {};
      const C_main = critCats.C || {};
      const D_main = directionCats.D || {};

      // Regime chip categories
      setText("chip-cat-C", C_main.category || "—");
      setText("chip-cat-S", S_main.category || "—");
      setText("chip-cat-K", K_main.category || "—");
      setText("chip-cat-D", D_main.category || "—");

      document.getElementById("chip-dot-C").className = "chip-dot " + categoryClass(C_main.category);
      document.getElementById("chip-dot-S").className = "chip-dot " + categoryClass(S_main.category);
      document.getElementById("chip-dot-K").className = "chip-dot " + categoryClass(K_main.category);
      document.getElementById("chip-dot-D").className = "chip-dot " + categoryClass(D_main.category);

      // Default explainer = criticality + its submetrics
      document.getElementById("regime-explainer").textContent =
        C_main.explanation || "System criticality explanation unavailable.";
      updateSubmetricPills("C");

      // Summary tab metrics
      function setMetric(rowId, obj) {
        const valueEl = document.getElementById(rowId + "-value");
        const badgeEl = document.getElementById(rowId + "-badge");
        const barEl = document.getElementById(rowId + "-bar");

        if (valueEl) {
          valueEl.textContent = obj.value != null ? obj.value.toFixed(2) : "—";
        }
        if (badgeEl) {
          const cls = categoryClass(obj.category);
          badgeEl.className = "metric-badge " + cls;
          badgeEl.innerHTML = `<span class="metric-badge-dot"></span>${obj.category || "—"}`;
        }
        if (barEl) {
          const v = typeof obj.value === "number" ? Math.max(0, Math.min(1, obj.value)) : null;
          barEl.style.width = v != null ? (v * 100).toFixed(0) + "%" : "0%";
        }
      }

      setMetric("metric-dryness", critCats.dryness_score || {});
      setMetric("metric-ivrv", critCats.ivrv_score || {});
      setMetric("metric-crowding", critCats.crowding_score || {});
      setMetric("metric-pcr", critCats.pcr_score || {});

      const shockScore = S_main.value;
      const contScore = K_main.value;
      const shockBadgeClass = categoryClass(S_main.category);
      const contBadgeClass = categoryClass(K_main.category);

      setText("metric-shock-score", shockScore != null ? shockScore.toFixed(2) : "—");
      setText("metric-contagion-score", contScore != null ? contScore.toFixed(2) : "—");

      const shockBadgeEl = document.getElementById("metric-shock-badge");
      const contBadgeEl = document.getElementById("metric-contagion-badge");
      const shockBarEl = document.getElementById("metric-shock-bar");
      const contBarEl = document.getElementById("metric-contagion-bar");

      if (shockBadgeEl) {
        shockBadgeEl.className = "metric-badge " + shockBadgeClass;
        shockBadgeEl.innerHTML = `<span class="metric-badge-dot"></span>${S_main.category || "—"}`;
      }
      if (contBadgeEl) {
        contBadgeEl.className = "metric-badge " + contBadgeClass;
        contBadgeEl.innerHTML = `<span class="metric-badge-dot"></span>${K_main.category || "—"}`;
      }
      if (shockBarEl) {
        const v = typeof shockScore === "number" ? Math.max(0, Math.min(1, shockScore)) : null;
        shockBarEl.style.width = v != null ? (v * 100).toFixed(0) + "%" : "0%";
      }
      if (contBarEl) {
        const v = typeof contScore === "number" ? Math.max(0, Math.min(1, contScore)) : null;
        contBarEl.style.width = v != null ? (v * 100).toFixed(0) + "%" : "0%";
      }

      // Intraday tab
      const intra = regimeMeta.intraday || {};
      setText("pill-mode", `Mode: ${intra.mode || "—"}`);
      setText("pill-meanrev", `Mean reversion: ${intra.enable_mean_reversion ? "ON" : "OFF"}`);
      setText("pill-breakout", `Breakout trend: ${intra.enable_breakout_trend ? "ON" : "OFF"}`);
      setText("pill-vol", `Vol: ${intra.volatility_scaling || "—"}`);
      setText("pill-bias", `Bias: ${intra.directional_bias || "—"}`);

      const notesUl = document.getElementById("intraday-notes");
      notesUl.innerHTML = "";
      (intra.notes || []).forEach((n) => {
        const li = document.createElement("li");
        li.textContent = n;
        notesUl.appendChild(li);
      });

      // Options tab
      const optMeta = regimeMeta.options || {};
      setText("pill-gamma-long", `Long gamma: ${optMeta.allow_long_gamma ? "Allowed" : "Avoid"}`);
      setText(
        "pill-gamma-short-intra",
        `Short gamma (intraday): ${optMeta.allow_short_gamma_intraday ? "Allowed" : "Avoid"}`
      );
      setText(
        "pill-gamma-short-overnight",
        `Short gamma (overnight): ${optMeta.allow_short_gamma_overnight ? "Allowed" : "Avoid"}`
      );

      setText("long-entry", longPlan.entry_spot != null ? longPlan.entry_spot.toFixed(1) : "—");
      setText("long-stop", longPlan.stop_spot != null ? longPlan.stop_spot.toFixed(1) : "—");
      setText(
        "long-targets",
        longPlan.target1_spot != null && longPlan.target2_spot != null
          ? `${longPlan.target1_spot.toFixed(1)} / ${longPlan.target2_spot.toFixed(1)}`
          : "—"
      );
      setText(
        "long-R",
        longPlan.R_multiple_target1 != null && longPlan.R_multiple_target2 != null
          ? `${longPlan.R_multiple_target1.toFixed(1)}R / ${longPlan.R_multiple_target2.toFixed(1)}R`
          : "—"
      );

      setText("short-entry", shortPlan.entry_spot != null ? shortPlan.entry_spot.toFixed(1) : "—");
      setText("short-stop", shortPlan.stop_spot != null ? shortPlan.stop_spot.toFixed(1) : "—");
      setText(
        "short-targets",
        shortPlan.target1_spot != null && shortPlan.target2_spot != null
          ? `${shortPlan.target1_spot.toFixed(1)} / ${shortPlan.target2_spot.toFixed(1)}`
          : "—"
      );
      setText(
        "short-R",
        shortPlan.R_multiple_target1 != null && shortPlan.R_multiple_target2 != null
          ? `${shortPlan.R_multiple_target1.toFixed(1)}R / ${shortPlan.R_multiple_target2.toFixed(1)}R`
          : "—"
      );

      const optNotesDiv = document.getElementById("options-notes");
      optNotesDiv.innerHTML = "";
      (optMeta.notes || []).forEach((n) => {
        const span = document.createElement("div");
        span.textContent = "• " + n;
        optNotesDiv.appendChild(span);
      });

      // Regime chip click -> explanation + submetrics swap
      const inputs = regime.inputs || {};
      const inputMap = {
        C: inputs.C || {},
        S: inputs.S || {},
        K: inputs.K || {},
        D: inputs.D || {},
      };
      const chipButtons = document.querySelectorAll(".regime-chip");
      chipButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
          chipButtons.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          const key = btn.getAttribute("data-regime");
          const info = inputMap[key] || {};
          document.getElementById("regime-explainer").textContent =
            info.explanation || "No explanation available for this component.";
          updateSubmetricPills(key);
        });
      });
    }

    async function loadSnapshot() {
      try {
        const res = await fetch(SNAPSHOT_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        applySnapshot(data);
      } catch (err) {
        console.error("Error loading snapshot:", err);
      }
    }

// EVENTS

let globalEvents = [];
let currentEventScope = "INR"; // INR = India, USD = Global

function parseEventDate(str) {
  if (!str) return null;
  // Expecting format like "12/01/2025 05:00:00"
  const parts = str.trim().split(" ");
  if (parts.length < 2) return null;
  const [d, m, y] = parts[0].split("/").map(Number);
  const [hh, mm, ss] = parts[1].split(":").map(Number);
  if (!d || !m || !y) return null;
  // Treat as UTC and adjust later to IST for display if needed
  return new Date(Date.UTC(y, m - 1, d, hh || 0, mm || 0, ss || 0));
}

function toISTfromDate(date) {
  if (!date || isNaN(date)) return "";
  const utcMs = date.getTime();
  const istOffsetMs = 5.5 * 60 * 60 * 1000;
  const istDate = new Date(utcMs + istOffsetMs);

  const day = String(istDate.getDate()).padStart(2, "0");
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const month = monthNames[istDate.getMonth()];
  const year = istDate.getFullYear();
  const hours = String(istDate.getHours()).padStart(2, "0");
  const mins = String(istDate.getMinutes()).padStart(2, "0");

  return `${day} ${month} ${year}, ${hours}:${mins} IST`;
}

function renderEvents(scope) {
  currentEventScope = scope || currentEventScope;
  const container = document.getElementById("events-list");
  if (!container) return;

  container.innerHTML = "";

  let filtered = globalEvents.filter(ev => {
    if (!ev.currency) return false;
    if (currentEventScope === "INR") return ev.currency === "INR";
    if (currentEventScope === "USD") return ev.currency === "USD";
    return true;
  });

  const now = new Date();
  filtered = filtered.filter(ev => ev.date && ev.date >= now);

  filtered.sort((a, b) => a.date - b.date);

  if (!filtered.length) {
    const div = document.createElement("div");
    div.className = "events-empty";
    div.textContent = "No upcoming events in this bucket.";
    container.appendChild(div);
    return;
  }

  filtered.slice(0, 5).forEach(ev => {
    const item = document.createElement("div");
    item.className = "events-item";

    const impactClass = (ev.impact || "").toLowerCase();
    const countryLabel = ev.currency === "INR" ? "India" : "Global";
    const countryEmoji = ev.currency === "INR" ? "🇮🇳" : "🌍";

    item.innerHTML = `
      <div class="events-row-top">
        <div class="events-name">${ev.name || "Unnamed event"}</div>
        <div class="events-time">${toISTfromDate(ev.date)}</div>
      </div>
      <div class="events-row-bottom">
        <span class="events-impact-pill ${impactClass}">${(ev.impact || "Medium").toUpperCase()}</span>
        <span class="events-country-pill">${countryEmoji}&nbsp;${countryLabel}</span>
      </div>
    `;

    container.appendChild(item);
  });
}

async function loadEvents() {
  try {
    const res = await fetch(EVENTS_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return;

    const headers = lines[0].split(",");
    const idxStart = headers.findIndex(h => h.trim().toLowerCase() === "start");
    const idxName = headers.findIndex(h => h.trim().toLowerCase() === "name");
    const idxImpact = headers.findIndex(h => h.trim().toLowerCase() === "impact");
    const idxCurrency = headers.findIndex(h => h.trim().toLowerCase() === "currency");

    globalEvents = lines.slice(1).map(line => {
      const cols = line.split(",");
      const dateStr = cols[idxStart] || "";
      return {
        date: parseEventDate(dateStr),
        name: cols[idxName] || "",
        impact: (cols[idxImpact] || "").toUpperCase(),
        currency: (cols[idxCurrency] || "").toUpperCase()
      };
    }).filter(ev => ev.date);

    renderEvents("INR");

    const buttons = document.querySelectorAll(".events-filter-button");
    buttons.forEach(btn => {
      btn.addEventListener("click", () => {
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const scope = btn.getAttribute("data-scope") || "INR";
        renderEvents(scope);
      });
    });
  } catch (err) {
    console.error("Error loading events:", err);
    const container = document.getElementById("events-list");
    if (container) {
      container.innerHTML = '<div class="events-empty">Error loading events.</div>';
    }
  }
}

    // NEWS

    function classifyNewsItem(item) {
      const title = (item.title || "").toLowerCase();
      const source = (item.source || "").toLowerCase();

      if (title.includes("gdp") || title.includes("inflation") || title.includes("rbi") || source.includes("forex")) {
        return "macro";
      }
      if (title.includes("sensex") || title.includes("nifty") || title.includes("index") || source.includes("nsei")) {
        return "index";
      }
      if (
        title.includes("ipo") ||
        title.includes("shares") ||
        title.includes("stock") ||
        title.includes("mutual fund") ||
        title.includes("broker")
      ) {
        return "stocks";
      }
      if (
        title.includes("forex") ||
        title.includes("rupee") ||
        title.includes("dollar") ||
        title.includes("gold") ||
        title.includes("silver") ||
        title.includes("crude") ||
        title.includes("brent")
      ) {
        return "fx";
      }
      if (
        title.includes("flows") ||
        title.includes("fii") ||
        title.includes("dii") ||
        title.includes("policy") ||
        title.includes("sebi")
      ) {
        return "flows";
      }
      return "other";
    }

    function toIST(dateStr) {
      if (!dateStr) return "";
      const d = new Date(dateStr);
      if (isNaN(d)) return dateStr;

      const utcMs = d.getTime();
      const istOffsetMs = 5.5 * 60 * 60 * 1000;
      const istDate = new Date(utcMs + istOffsetMs);

      const day = String(istDate.getDate()).padStart(2, "0");
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const month = monthNames[istDate.getMonth()];
      const year = istDate.getFullYear();
      const hours = String(istDate.getHours()).padStart(2, "0");
      const mins = String(istDate.getMinutes()).padStart(2, "0");

      return `${day} ${month} ${year}, ${hours}:${mins} IST`;
    }

    let globalNewsItems = [];

    function renderNews(filter) {
      const container = document.getElementById("news-list");
      container.innerHTML = "";

      const items = globalNewsItems.filter((item) => {
        const bucket = classifyNewsItem(item);
        if (filter === "all") return true;
        if (filter === "macro") return bucket === "macro";
        if (filter === "index") return bucket === "index";
        if (filter === "stocks") return bucket === "stocks";
        if (filter === "fx") return bucket === "fx";
        if (filter === "flows") return bucket === "flows";
        return true;
      });

      if (!items.length) {
        const div = document.createElement("div");
        div.className = "news-empty";
        div.textContent = "No news items for this filter.";
        container.appendChild(div);
        return;
      }

      items.slice(0, 18).forEach((n) => {
        const bucket = classifyNewsItem(n);
        const article = document.createElement("article");
        article.className = "news-item";

        const tagLabel =
          bucket === "macro"
            ? "Macro"
            : bucket === "index"
            ? "Index"
            : bucket === "stocks"
            ? "Stocks"
            : bucket === "fx"
            ? "FX / Commodities"
            : bucket === "flows"
            ? "Flows / Policy"
            : "Other";

        const score = typeof n.sentiment_score === "number" ? n.sentiment_score : null;
        const label = (n.sentiment_label || "neutral").toLowerCase();
        const mag = score != null ? Math.round(Math.abs(score) * 100) / 10 : null; // 0.95 -> 9.5
        const magText = mag != null ? mag.toFixed(1) : "—";
        const sentimentClass =
          label === "positive" ? "positive" :
          label === "negative" ? "negative" :
          "neutral";

        const dateIST = n.published ? toIST(n.published) : "";

        article.innerHTML = `
          <div class="news-title-row">
            <span class="news-tag">${tagLabel}</span>
            <div class="news-title">
              <a href="${n.link || "#"}" target="_blank" rel="noopener noreferrer">${n.title || "Untitled"}</a>
            </div>
          </div>
          <div class="news-meta">
            ${n.source || ""}${dateIST ? " · " + dateIST : ""}
          </div>
          <div class="news-sentiment">
            <div class="sentiment-score-circle ${sentimentClass}">
              ${magText}
            </div>
            <span class="sentiment-pill ${sentimentClass}">
              ${(label || "neutral").toUpperCase()}
            </span>
          </div>
        `;

        container.appendChild(article);
      });
    }

    async function loadNews() {
      try {
        const res = await fetch(NEWS_URL, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const items = data.items || [];

        // sort by published date (newest first)
        globalNewsItems = items.slice().sort((a, b) => {
          const da = new Date(a.published);
          const db = new Date(b.published);
          if (isNaN(da) && isNaN(db)) return 0;
          if (isNaN(da)) return 1;
          if (isNaN(db)) return -1;
          return db - da;
        });

        renderNews("all");

        const filters = document.querySelectorAll(".news-filter");
        filters.forEach((btn) => {
          btn.addEventListener("click", () => {
            filters.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            const f = btn.getAttribute("data-filter");
            renderNews(f);
          });
        });
      } catch (err) {
        console.error("Error loading news:", err);
        const container = document.getElementById("news-list");
        container.innerHTML = '<div class="news-empty">Error loading news feed.</div>';
      }
    }

    // TABS INIT
    function initTabs() {
      const tabButtons = document.querySelectorAll(".tab-button");
      const panels = {
        summary: document.getElementById("tab-summary"),
        intraday: document.getElementById("tab-intraday"),
        options: document.getElementById("tab-options"),
      };

      tabButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
          const tab = btn.getAttribute("data-tab");
          tabButtons.forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");

          Object.entries(panels).forEach(([key, panel]) => {
            panel.classList.toggle("active", key === tab);
          });
        });
      });
    }

    // DEFINITIONS TOGGLE
    function initDefinitions() {
      const toggle = document.getElementById("defs-toggle");
      const panel = document.getElementById("defs-panel");
      if (!toggle || !panel) return;

      toggle.addEventListener("click", () => {
        panel.classList.toggle("open");
      });
    }

    (function init() {
      initTabs();
      initDefinitions();
      loadSnapshot();
      loadNews();
      loadEvents();
    })();
  
