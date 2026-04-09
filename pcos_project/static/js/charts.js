(() => {
    const app = window.PCOSApp;
    const board = document.getElementById("analysisBoard");
    const emptyState = document.getElementById("analysisEmptyState");
    if (!app || !board) {
        return;
    }

    const DEFAULT_ANALYSIS = {
        has_prediction: false,
        message: "Run a prediction to view personalized PCOS analysis charts.",
        mode: "basic",
        mode_label: "Basic Screening",
        risk_probability: {
            labels: ["Predicted risk", "Remaining"],
            values: [0, 100],
            probability: 0,
        },
        symptom_contribution: {
            labels: [],
            values: [],
        },
        bmi_vs_risk: {
            points: [],
            patient: { x: 0, y: 0 },
        },
        age_vs_risk: {
            points: [],
            patient: { x: 0, y: 0 },
        },
    };

    const charts = {};

    function cloneDefaults() {
        return JSON.parse(JSON.stringify(DEFAULT_ANALYSIS));
    }

    function normalizeRiskProbability(value) {
        if (typeof value === "number") {
            const safeValue = Math.max(0, Math.min(1, value));
            return {
                labels: ["Predicted risk", "Remaining"],
                values: [safeValue * 100, (1 - safeValue) * 100],
                probability: safeValue,
            };
        }

        const labels = Array.isArray(value?.labels) && value.labels.length === 2
            ? value.labels
            : ["Predicted risk", "Remaining"];
        const values = Array.isArray(value?.values) && value.values.length === 2
            ? value.values.map((entry) => Number(entry) || 0)
            : [0, 100];
        const probability = typeof value?.probability === "number"
            ? value.probability
            : Math.max(0, Math.min(1, (values[0] || 0) / 100));

        return { labels, values, probability };
    }

    function normalizeContribution(value) {
        if (!value) {
            return { labels: [], values: [] };
        }

        if (Array.isArray(value?.labels) && Array.isArray(value?.values)) {
            return {
                labels: value.labels,
                values: value.values.map((entry) => Number(entry) || 0),
            };
        }

        if (typeof value === "object") {
            return {
                labels: Object.keys(value),
                values: Object.values(value).map((entry) => Number(entry) || 0),
            };
        }

        return { labels: [], values: [] };
    }

    function normalizeLineData(value) {
        return {
            points: Array.isArray(value?.points)
                ? value.points.map((point) => ({
                    x: Number(point?.x) || 0,
                    y: Number(point?.y) || 0,
                }))
                : [],
            patient: {
                x: Number(value?.patient?.x) || 0,
                y: Number(value?.patient?.y) || 0,
            },
        };
    }

    function normalizeAnalysisPayload(payload) {
        const base = cloneDefaults();
        const candidate = payload?.analysis ?? payload ?? {};

        base.has_prediction = Boolean(candidate?.has_prediction);
        base.message = candidate?.message ?? base.message;
        base.mode = candidate?.mode ?? base.mode;
        base.mode_label = candidate?.mode_label ?? base.mode_label;
        base.risk_probability = normalizeRiskProbability(candidate?.risk_probability);
        base.symptom_contribution = normalizeContribution(
            candidate?.feature_importance ?? candidate?.symptom_contribution ?? candidate?.symptoms
        );
        base.bmi_vs_risk = normalizeLineData(candidate?.bmi_vs_risk);
        base.age_vs_risk = normalizeLineData(candidate?.age_vs_risk);
        return base;
    }

    function createCharts() {
        const riskCanvas = document.getElementById("analysisRiskChart");
        const symptomCanvas = document.getElementById("symptomContributionChart");
        const bmiCanvas = document.getElementById("bmiRiskChart");
        const ageCanvas = document.getElementById("ageRiskChart");
        if (!riskCanvas || !symptomCanvas || !bmiCanvas || !ageCanvas) {
            return;
        }

        charts.risk = new Chart(riskCanvas, {
            type: "doughnut",
            data: {
                labels: ["Predicted risk", "Remaining"],
                datasets: [
                    {
                        data: [0, 100],
                        backgroundColor: ["#2f80ed", "#e8eef8"],
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                cutout: "72%",
                animation: { duration: 1100, easing: "easeOutQuart" },
                plugins: { legend: { position: "bottom" } },
            },
        });

        charts.symptom = new Chart(symptomCanvas, {
            type: "bar",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "AI explanation",
                        data: [],
                        backgroundColor: "#2f80ed",
                        borderRadius: 10,
                    },
                ],
            },
            options: {
                animation: { duration: 1200, easing: "easeOutQuart" },
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, suggestedMax: 100, title: { display: true, text: "Relative contribution" } },
                },
            },
        });

        charts.bmi = new Chart(bmiCanvas, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "BMI risk curve",
                        data: [],
                        parsing: false,
                        borderColor: "#27ae60",
                        backgroundColor: "rgba(39, 174, 96, 0.12)",
                        fill: true,
                        tension: 0.35,
                    },
                    {
                        label: "Current user",
                        data: [{ x: 0, y: 0 }],
                        parsing: false,
                        pointRadius: 6,
                        pointHoverRadius: 7,
                        showLine: false,
                        backgroundColor: "#e74c3c",
                    },
                ],
            },
            options: {
                animation: { duration: 1200, easing: "easeOutQuart" },
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: { type: "linear", title: { display: true, text: "BMI" } },
                    y: { title: { display: true, text: "Observed PCOS risk (%)" }, beginAtZero: true, suggestedMax: 100 },
                },
            },
        });

        charts.age = new Chart(ageCanvas, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Age risk curve",
                        data: [],
                        parsing: false,
                        borderColor: "#f39c12",
                        backgroundColor: "rgba(243, 156, 18, 0.12)",
                        fill: true,
                        tension: 0.35,
                    },
                    {
                        label: "Current user",
                        data: [{ x: 0, y: 0 }],
                        parsing: false,
                        pointRadius: 6,
                        pointHoverRadius: 7,
                        showLine: false,
                        backgroundColor: "#e74c3c",
                    },
                ],
            },
            options: {
                animation: { duration: 1200, easing: "easeOutQuart" },
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: { type: "linear", title: { display: true, text: "Age" } },
                    y: { title: { display: true, text: "Observed PCOS risk (%)" }, beginAtZero: true, suggestedMax: 100 },
                },
            },
        });
    }

    function updateRiskChart(probabilityData) {
        if (!charts.risk) {
            return;
        }
        charts.risk.data.labels = probabilityData.labels;
        charts.risk.data.datasets[0].data = probabilityData.values;
        charts.risk.update();
    }

    function updateSymptomChart(symptoms, label) {
        if (!charts.symptom) {
            return;
        }
        const labels = symptoms?.labels ?? [];
        const values = symptoms?.values ?? [];
        charts.symptom.data.labels = labels;
        charts.symptom.data.datasets[0].label = label ? `${label} explanation` : "AI explanation";
        charts.symptom.data.datasets[0].data = values;
        charts.symptom.update();
    }

    function updateLineChart(chart, lineData) {
        if (!chart) {
            return;
        }
        chart.data.datasets[0].data = lineData.points;
        chart.data.datasets[1].data = [lineData.patient];
        chart.update();
    }

    function toggleEmptyState(analysis) {
        if (!emptyState) {
            return;
        }
        const shouldShow = !analysis.has_prediction && (analysis.risk_probability?.probability ?? 0) === 0;
        emptyState.classList.toggle("d-none", !shouldShow);
        if (shouldShow) {
            const alert = emptyState.querySelector(".alert");
            if (alert) {
                alert.textContent = analysis.message || DEFAULT_ANALYSIS.message;
            }
        }
    }

    function renderAnalytics(payload) {
        const analysis = normalizeAnalysisPayload(payload);
        updateRiskChart(analysis.risk_probability);
        updateSymptomChart(analysis.symptom_contribution, analysis.mode_label);
        updateLineChart(charts.bmi, analysis.bmi_vs_risk);
        updateLineChart(charts.age, analysis.age_vs_risk);
        toggleEmptyState(analysis);
    }

    async function loadAnalytics() {
        try {
            const response = await app.fetchJSON(board.dataset.endpoint);
            renderAnalytics(response);
        } catch (error) {
            console.error("Analysis data error:", error);
            renderAnalytics(DEFAULT_ANALYSIS);
            const existing = board.querySelector(".status-banner");
            if (existing) {
                existing.remove();
            }
            const message = document.createElement("div");
            message.className = "status-banner";
            message.innerHTML = `<div class="alert alert-danger mb-0">${error.message || "Analytics could not be loaded."}</div>`;
            board.prepend(message);
        }
    }

    const refreshBtn = document.getElementById("refreshAnalyticsBtn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", loadAnalytics);
    }

    window.addEventListener("prediction:updated", (event) => {
        renderAnalytics(event?.detail ?? DEFAULT_ANALYSIS);
    });

    createCharts();
    renderAnalytics(DEFAULT_ANALYSIS);
    loadAnalytics();
})();
