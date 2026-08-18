(() => {
    "use strict";

    const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"];

    const el = {
        loading: document.getElementById("loading-overlay"),
        errorBanner: document.getElementById("error-banner"),
        appBody: document.getElementById("app-body"),
        commodityGroup: document.getElementById("commodity-group"),
        zoneSelect: document.getElementById("zone-select"),
        headerCommodity: document.getElementById("header-commodity"),
        headerZone: document.getElementById("header-zone"),
        metricN: document.getElementById("metric-n"),
        metricR2: document.getElementById("metric-r2"),
        metricRmse: document.getElementById("metric-rmse"),
        dateStart: document.getElementById("date-start"),
        dateEnd: document.getElementById("date-end"),
        autoscale: document.getElementById("autoscale"),
        historyError: document.getElementById("history-error"),
        chartCanvas: document.getElementById("history-chart"),
        monthSelect: document.getElementById("month-select"),
        curveError: document.getElementById("curve-error"),
        curveDemandUnit: document.getElementById("curve-demand-unit"),
        curvePriceUnit: document.getElementById("curve-price-unit"),
        demandCurveCanvas: document.getElementById("demand-curve-chart"),
        priceCurveCanvas: document.getElementById("price-curve-chart"),
        resetCurveBtn: document.getElementById("reset-curve-btn"),
        hourSteppers: document.getElementById("hour-steppers"),
    };

    const DEMAND_BOUND_PCT = 0.3;
    const DEMAND_BOUND_IDX = { upper: 0, lower: 1, representative: 2, modified: 3 };
    const DEMAND_STEP_MW = 20;

    let historyChart = null;
    let historyToken = 0;

    let demandChart = null;
    let priceChart = null;
    let curveToken = 0;
    let demandUnit = "";
    let priceUnit = "";
    let demandValues = [];
    let originalValues = [];
    let draggingIndex = null;
    let hourStepperEls = [];
    let hourDeltaEls = [];

    function showLoading(on) {
        el.loading.classList.toggle("hidden", !on);
    }

    function showError(msg) {
        el.errorBanner.textContent = msg;
        el.errorBanner.classList.remove("hidden");
        el.appBody.classList.add("hidden");
    }

    function hideError() {
        el.errorBanner.classList.add("hidden");
        el.appBody.classList.remove("hidden");
        if (historyChart) historyChart.resize();
        if (demandChart) demandChart.resize();
        if (priceChart) priceChart.resize();
    }

    function showHistoryError(msg) {
        el.historyError.textContent = msg;
        el.historyError.classList.remove("hidden");
    }

    function hideHistoryError() {
        el.historyError.classList.add("hidden");
    }

    function showCurveError(msg) {
        el.curveError.textContent = msg;
        el.curveError.classList.remove("hidden");
    }

    function hideCurveError() {
        el.curveError.classList.add("hidden");
    }

    async function getJSON(url) {
        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || `Request failed (${res.status})`);
        }
        return data;
    }

    function currentCommodity() {
        return el.commodityGroup.querySelector('input[name="commodity"]:checked').value;
    }

    function currentMonth() {
        return parseInt(el.monthSelect.value, 10);
    }

    function populateMonthSelect() {
        el.monthSelect.innerHTML = "";
        MONTH_NAMES.forEach((name, i) => {
            const opt = document.createElement("option");
            opt.value = String(i + 1);
            opt.textContent = name;
            el.monthSelect.appendChild(opt);
        });
        el.monthSelect.value = "1";
    }

    function initHistoryChart() {
        historyChart = new Chart(el.chartCanvas.getContext("2d"), {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "PLEXOS",
                        data: [],
                        borderColor: "#2a78d6",
                        backgroundColor: "#2a78d6",
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0,
                    },
                    {
                        label: "Emulated",
                        data: [],
                        borderColor: "#eb6834",
                        backgroundColor: "#eb6834",
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        type: "time",
                        time: {
                            tooltipFormat: "MMM d, yyyy HH:mm",
                            displayFormats: {
                                hour: "MMM d, HH:mm",
                                day: "MMM d",
                                week: "MMM d",
                                month: "MMM yyyy",
                            },
                        },
                        ticks: {
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10,
                        },
                        title: { display: true, text: "Date" },
                    },
                    y: {
                        title: { display: true, text: "Price" },
                        beginAtZero: true,
                    },
                },
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}`,
                        },
                    },
                },
            },
        });
    }

    function setYScale(unit, autoscale, values) {
        const finite = values.filter((v) => typeof v === "number" && Number.isFinite(v));
        if (autoscale && finite.length) {
            const min = Math.min(...finite);
            const max = Math.max(...finite);
            const pad = (max - min) * 0.05 || 1.0;
            historyChart.options.scales.y.min = min - pad;
            historyChart.options.scales.y.max = max + pad;
            historyChart.options.scales.y.beginAtZero = false;
        } else {
            historyChart.options.scales.y.min = undefined;
            historyChart.options.scales.y.max = undefined;
            historyChart.options.scales.y.beginAtZero = true;
        }
        historyChart.options.scales.y.title.text = unit ? `Price [${unit}]` : "Price";
    }

    async function refreshHistory(commodity, zone) {
        const token = ++historyToken;
        const start = el.dateStart.value;
        const end = el.dateEnd.value;
        if (!start || !end) return;

        hideHistoryError();
        try {
            const data = await getJSON(
                `/api/history/${commodity}/${zone}/?start=${start}&end=${end}`
            );
            if (token !== historyToken) return;

            const actualPoints = data.datetime.map((d, i) => ({ x: d, y: data.actual[i] }));
            const emulatedPoints = data.datetime.map((d, i) => ({ x: d, y: data.emulated[i] }));
            historyChart.data.datasets[0].data = actualPoints;
            historyChart.data.datasets[1].data = emulatedPoints;
            setYScale(data.unit, el.autoscale.checked, [...data.actual, ...data.emulated]);
            historyChart.update();
        } catch (err) {
            if (token !== historyToken) return;
            showHistoryError(err.message);
        }
    }

    function hourLineOptions(titleText, tooltipUnit, showLegend) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: showLegend ? { mode: "index", intersect: false } : undefined,
            scales: {
                x: {
                    type: "linear",
                    min: -0.5,
                    max: 23.5,
                    afterBuildTicks: (axis) => {
                        axis.ticks = Array.from({ length: 24 }, (_, h) => ({ value: h }));
                    },
                    ticks: { callback: (v) => `${v + 1}` },
                    title: { display: true, text: "Hour" },
                },
                y: {
                    title: { display: true, text: titleText },
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: {
                    display: !!showLegend,
                    position: "top",
                    labels: { filter: (item, data) => !data.datasets[item.datasetIndex].isBound },
                },
                tooltip: {
                    filter: (item) => !item.dataset.isBound,
                    callbacks: {
                        title: (items) => `Hour ${items[0].parsed.x + 1}`,
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} ${tooltipUnit()}`,
                    },
                },
            },
        };
    }

    function initCurveCharts() {
        demandChart = new Chart(el.demandCurveCanvas.getContext("2d"), {
            type: "line",
            data: {
                datasets: [
                    {
                        label: `+${DEMAND_BOUND_PCT * 100}% bound`,
                        data: [],
                        isBound: true,
                        borderWidth: 0,
                        pointRadius: 0,
                        pointHitRadius: 0,
                        tension: 0.25,
                        fill: DEMAND_BOUND_IDX.lower,
                        backgroundColor: "rgba(120, 120, 120, 0.18)",
                    },
                    {
                        label: `-${DEMAND_BOUND_PCT * 100}% bound`,
                        data: [],
                        isBound: true,
                        borderWidth: 0,
                        pointRadius: 0,
                        pointHitRadius: 0,
                        tension: 0.25,
                        fill: false,
                    },
                    {
                        label: "Representative",
                        data: [],
                        borderColor: "#6b6b6b",
                        backgroundColor: "#6b6b6b",
                        borderWidth: 2,
                        borderDash: [6, 4],
                        pointRadius: 2,
                        tension: 0.25,
                    },
                    {
                        label: "Modified",
                        data: [],
                        borderColor: "#2a78d6",
                        backgroundColor: "#2a78d6",
                        borderWidth: 2,
                        pointRadius: 6,
                        pointHitRadius: 14,
                        pointHoverRadius: 8,
                        tension: 0.25,
                    },
                ],
            },
            options: hourLineOptions("Demand", () => demandUnit, true),
        });

        priceChart = new Chart(el.priceCurveCanvas.getContext("2d"), {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Representative",
                        data: [],
                        borderColor: "#6b6b6b",
                        backgroundColor: "#6b6b6b",
                        borderWidth: 2,
                        borderDash: [6, 4],
                        pointRadius: 2,
                        tension: 0.25,
                    },
                    {
                        label: "Modified",
                        data: [],
                        borderColor: "#eb6834",
                        backgroundColor: "#eb6834",
                        borderWidth: 2,
                        pointRadius: 3,
                        tension: 0.25,
                    },
                ],
            },
            options: hourLineOptions("Price", () => priceUnit, true),
        });

        wireDrag();

        new ResizeObserver(() => {
            requestAnimationFrame(() => requestAnimationFrame(positionHourSteppers));
        }).observe(el.demandCurveCanvas.parentElement);
    }

    function setDemandChartDataset(index, values) {
        demandChart.data.datasets[index].data = values.map((v, h) => ({ x: h, y: v }));
        demandChart.update("none");
        positionHourSteppers();
    }

    function demandBoundAt(hour) {
        const rep = originalValues[hour] ?? 0;
        return {
            lower: Math.max(0, rep * (1 - DEMAND_BOUND_PCT)),
            upper: rep * (1 + DEMAND_BOUND_PCT),
        };
    }

    function setDemandBoundDatasets() {
        const upper = originalValues.map((v) => Math.max(0, v) * (1 + DEMAND_BOUND_PCT));
        const lower = originalValues.map((v) => Math.max(0, v) * (1 - DEMAND_BOUND_PCT));
        setDemandChartDataset(DEMAND_BOUND_IDX.upper, upper);
        setDemandChartDataset(DEMAND_BOUND_IDX.lower, lower);
    }

    function autoscaleAxis(chart, values, includeZero) {
        const finite = values.filter((v) => typeof v === "number" && Number.isFinite(v));
        if (!finite.length) return;
        let min = Math.min(...finite);
        let max = Math.max(...finite);
        if (includeZero) {
            min = Math.min(min, 0);
            max = Math.max(max, 0);
        }
        const pad = (max - min) * 0.08 || 1.0;
        chart.options.scales.y.min = min - pad;
        chart.options.scales.y.max = max + pad;
    }

    function setPriceChartDataset(index, values) {
        priceChart.data.datasets[index].data = values.map((v, h) => ({ x: h, y: v }));
        const combined = priceChart.data.datasets.flatMap((ds) => ds.data.map((p) => p.y));
        autoscaleAxis(priceChart, combined, false);
        priceChart.update("none");
    }

    function eventPoint(evt) {
        if (evt.touches && evt.touches.length) return evt.touches[0];
        if (evt.changedTouches && evt.changedTouches.length) return evt.changedTouches[0];
        return evt;
    }

    function nearestHourIndex(clientX) {
        const rect = el.demandCurveCanvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const xScale = demandChart.scales.x;
        const spacing = Math.abs(xScale.getPixelForValue(1) - xScale.getPixelForValue(0));
        let best = 0;
        let bestDist = Infinity;
        for (let h = 0; h < 24; h++) {
            const dist = Math.abs(xScale.getPixelForValue(h) - x);
            if (dist < bestDist) {
                bestDist = dist;
                best = h;
            }
        }
        return bestDist <= spacing / 2 ? best : null;
    }

    function valueForClientY(clientY) {
        const rect = el.demandCurveCanvas.getBoundingClientRect();
        const y = clientY - rect.top;
        return Math.max(0, demandChart.scales.y.getValueForPixel(y));
    }

    function onDragStart(evt) {
        const p = eventPoint(evt);
        const idx = nearestHourIndex(p.clientX);
        if (idx === null) return;
        draggingIndex = idx;
        evt.preventDefault();
        onDragMove(evt);
    }

    function onDragMove(evt) {
        if (draggingIndex === null) return;
        const p = eventPoint(evt);
        const raw = valueForClientY(p.clientY);
        const { lower, upper } = demandBoundAt(draggingIndex);
        const value = Math.min(upper, Math.max(lower, raw));
        demandValues[draggingIndex] = value;
        demandChart.data.datasets[DEMAND_BOUND_IDX.modified].data[draggingIndex] = { x: draggingIndex, y: value };
        demandChart.update("none");
        updateHourValueLabels();
        evt.preventDefault();
    }

    function commitDemandCurve() {
        const commodity = currentCommodity();
        const zone = el.zoneSelect.value;
        const month = currentMonth();
        fetchModifiedPriceCurve(commodity, zone, month, demandValues);
    }

    function onDragEnd() {
        if (draggingIndex === null) return;
        draggingIndex = null;
        commitDemandCurve();
    }

    function buildHourSteppers() {
        el.hourSteppers.innerHTML = "";
        hourStepperEls = [];
        hourDeltaEls = [];
        for (let h = 0; h < 24; h++) {
            const wrap = document.createElement("div");
            wrap.className = "hour-stepper";

            const delta = document.createElement("span");
            delta.className = "hour-delta";
            delta.textContent = "0";

            const plus = document.createElement("button");
            plus.type = "button";
            plus.className = "step-btn";
            plus.textContent = "+";
            plus.setAttribute("aria-label", `Increase hour ${h + 1} by ${DEMAND_STEP_MW} MW`);
            plus.addEventListener("click", () => stepDemand(h, DEMAND_STEP_MW));

            const label = document.createElement("span");
            label.className = "hour-label";
            label.textContent = String(h + 1);

            const minus = document.createElement("button");
            minus.type = "button";
            minus.className = "step-btn";
            minus.textContent = "−";
            minus.setAttribute("aria-label", `Decrease hour ${h + 1} by ${DEMAND_STEP_MW} MW`);
            minus.addEventListener("click", () => stepDemand(h, -DEMAND_STEP_MW));

            wrap.append(delta, plus, label, minus);
            el.hourSteppers.appendChild(wrap);
            hourStepperEls.push(wrap);
            hourDeltaEls.push(delta);
        }
        positionHourSteppers();
    }

    function positionHourSteppers() {
        if (!demandChart || !demandChart.scales || !demandChart.scales.x) return;
        const xScale = demandChart.scales.x;
        hourStepperEls.forEach((wrap, h) => {
            wrap.style.left = `${xScale.getPixelForValue(h)}px`;
        });
    }

    function updateHourValueLabels() {
        demandValues.forEach((v, h) => {
            const delta = Math.round(v - (originalValues[h] ?? v));
            if (hourDeltaEls[h]) {
                hourDeltaEls[h].textContent = delta > 0 ? `+${delta}` : String(delta);
                hourDeltaEls[h].classList.toggle("positive", delta > 0);
                hourDeltaEls[h].classList.toggle("negative", delta < 0);
            }
            if (hourStepperEls[h]) {
                hourStepperEls[h].title = `Hour ${h + 1}: ${Math.round(v).toLocaleString()} ${demandUnit}`;
            }
        });
    }

    function stepDemand(hour, delta) {
        if (!demandValues.length) return;
        const { lower, upper } = demandBoundAt(hour);
        const value = Math.min(upper, Math.max(lower, demandValues[hour] + delta));
        demandValues[hour] = value;
        demandChart.data.datasets[DEMAND_BOUND_IDX.modified].data[hour] = { x: hour, y: value };
        demandChart.update("none");
        updateHourValueLabels();
        commitDemandCurve();
    }

    function wireDrag() {
        el.demandCurveCanvas.addEventListener("mousedown", onDragStart);
        document.addEventListener("mousemove", onDragMove);
        document.addEventListener("mouseup", onDragEnd);
        el.demandCurveCanvas.addEventListener("touchstart", onDragStart, { passive: false });
        document.addEventListener("touchmove", onDragMove, { passive: false });
        document.addEventListener("touchend", onDragEnd);
    }

    async function fetchPrice(commodity, zone, month, values) {
        const params = new URLSearchParams({ month: String(month), demand: values.join(",") });
        return getJSON(`/api/monthly-price-curve/${commodity}/${zone}/?${params}`);
    }

    async function fetchModifiedPriceCurve(commodity, zone, month, values) {
        const token = ++curveToken;
        try {
            const data = await fetchPrice(commodity, zone, month, values);
            if (token !== curveToken) return;
            priceUnit = data.unit;
            el.curvePriceUnit.textContent = priceUnit;
            setPriceChartDataset(1, data.price);
            hideCurveError();
        } catch (err) {
            if (token !== curveToken) return;
            showCurveError(err.message);
        }
    }

    async function loadCurve(commodity, zone, month) {
        try {
            const data = await getJSON(`/api/monthly-demand-curve/${commodity}/${zone}/?month=${month}`);
            hideCurveError();
            demandUnit = data.unit;
            el.curveDemandUnit.textContent = data.unit;
            demandValues = data.demand.slice();
            originalValues = data.demand.slice();
            setDemandBoundDatasets();
            setDemandChartDataset(DEMAND_BOUND_IDX.representative, originalValues);
            setDemandChartDataset(DEMAND_BOUND_IDX.modified, demandValues);
            updateHourValueLabels();

            const token = ++curveToken;
            const priceData = await fetchPrice(commodity, zone, month, originalValues);
            if (token !== curveToken) return;
            priceUnit = priceData.unit;
            el.curvePriceUnit.textContent = priceUnit;
            setPriceChartDataset(0, priceData.price);
            setPriceChartDataset(1, priceData.price);
        } catch (err) {
            showCurveError(err.message);
        }
    }

    function resetCurve() {
        if (!originalValues.length) return;
        demandValues = originalValues.slice();
        setDemandChartDataset(DEMAND_BOUND_IDX.modified, demandValues);
        updateHourValueLabels();
        setPriceChartDataset(1, priceChart.data.datasets[0].data.map((p) => p.y));
    }

    async function loadZone(commodity, zone, resetDates) {
        showLoading(true);
        try {
            const metrics = await getJSON(`/api/metrics/${commodity}/${zone}/`);
            hideError();

            el.headerCommodity.textContent = `Commodity: ${metrics.commodity_label}`;
            el.headerZone.textContent = `Zone: ${metrics.zone_label}`;
            el.metricN.textContent = metrics.n.toLocaleString();
            el.metricR2.textContent = metrics.cv_r2.toFixed(3);
            el.metricRmse.textContent = `${metrics.cv_rmse.toFixed(2)} ${metrics.unit}`;

            if (resetDates) {
                el.dateStart.min = metrics.date_bounds.min;
                el.dateStart.max = metrics.date_bounds.max;
                el.dateEnd.min = metrics.date_bounds.min;
                el.dateEnd.max = metrics.date_bounds.max;
                el.dateStart.value = metrics.date_bounds.min;
                el.dateEnd.value = metrics.date_bounds.default_end;
            }

            await refreshHistory(commodity, zone);
            await loadCurve(commodity, zone, currentMonth());
        } catch (err) {
            showError(err.message);
        } finally {
            showLoading(false);
        }
    }

    async function loadZonesForCommodity(commodity) {
        showLoading(true);
        try {
            const data = await getJSON(`/api/zones/${commodity}/`);
            hideError();
            el.zoneSelect.innerHTML = "";
            for (const z of data.zones) {
                const opt = document.createElement("option");
                opt.value = z.code;
                opt.textContent = z.label;
                el.zoneSelect.appendChild(opt);
            }
            el.zoneSelect.value = data.default;
            await loadZone(commodity, data.default, true);
        } catch (err) {
            showError(err.message);
            showLoading(false);
        }
    }

    el.commodityGroup.addEventListener("change", () => {
        loadZonesForCommodity(currentCommodity());
    });

    el.zoneSelect.addEventListener("change", () => {
        loadZone(currentCommodity(), el.zoneSelect.value, false);
    });

    el.dateStart.addEventListener("change", () => refreshHistory(currentCommodity(), el.zoneSelect.value));
    el.dateEnd.addEventListener("change", () => refreshHistory(currentCommodity(), el.zoneSelect.value));
    el.autoscale.addEventListener("change", () => refreshHistory(currentCommodity(), el.zoneSelect.value));

    el.monthSelect.addEventListener("change", () =>
        loadCurve(currentCommodity(), el.zoneSelect.value, currentMonth()));
    el.resetCurveBtn.addEventListener("click", resetCurve);

    populateMonthSelect();
    initHistoryChart();
    initCurveCharts();
    buildHourSteppers();
    loadZonesForCommodity(currentCommodity());
})();
