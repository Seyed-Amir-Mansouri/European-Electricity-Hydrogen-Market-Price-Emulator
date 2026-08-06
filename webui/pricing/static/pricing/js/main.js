(() => {
    "use strict";

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
    };

    let chart = null;
    let historyToken = 0; // guards against out-of-order fetch responses

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
    }

    function showHistoryError(msg) {
        el.historyError.textContent = msg;
        el.historyError.classList.remove("hidden");
    }

    function hideHistoryError() {
        el.historyError.classList.add("hidden");
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

    function initChart() {
        chart = new Chart(el.chartCanvas.getContext("2d"), {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Actual",
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
                parsing: false,
                scales: {
                    x: {
                        type: "time",
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
            chart.options.scales.y.min = min - pad;
            chart.options.scales.y.max = max + pad;
            chart.options.scales.y.beginAtZero = false;
        } else {
            chart.options.scales.y.min = undefined;
            chart.options.scales.y.max = undefined;
            chart.options.scales.y.beginAtZero = true;
        }
        chart.options.scales.y.title.text = unit ? `Price [${unit}]` : "Price";
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
            if (token !== historyToken) return; // a newer request has since started

            const actualPoints = data.datetime.map((d, i) => ({ x: d, y: data.actual[i] }));
            const emulatedPoints = data.datetime.map((d, i) => ({ x: d, y: data.emulated[i] }));
            chart.data.datasets[0].data = actualPoints;
            chart.data.datasets[1].data = emulatedPoints;
            setYScale(data.unit, el.autoscale.checked, [...data.actual, ...data.emulated]);
            chart.update();
        } catch (err) {
            if (token !== historyToken) return;
            showHistoryError(err.message);
        }
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

    initChart();
    loadZonesForCommodity(currentCommodity());
})();
