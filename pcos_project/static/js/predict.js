(() => {
    const app = window.PCOSApp;
    if (!app) {
        return;
    }

    let gaugeChart = null;

    const form = document.getElementById("predictionForm");
    const messageTarget = document.getElementById("predictMessage");
    const progressBar = document.getElementById("stepProgressBar");
    const progressLabel = document.getElementById("stepProgressLabel");
    const progressPercent = document.getElementById("stepProgressPercent");
    const nextBtn = document.getElementById("nextStepBtn");
    const prevBtn = document.getElementById("prevStepBtn");
    const submitBtn = document.getElementById("submitPredictionBtn");
    const modeInput = document.getElementById("predictionModeInput");
    const modeRadios = document.querySelectorAll(".mode-radio");
    const bmiValue = document.getElementById("calculatedBmiValue");
    const heightInput = document.getElementById("field_height_cm");
    const weightInput = document.getElementById("field_weight_kg");
    const steps = Array.from(document.querySelectorAll(".form-step"));
    const indicators = Array.from(document.querySelectorAll("[data-step-indicator]"));

    let currentStepIndex = 0;

    function getRiskGaugeConfig(probabilityPct) {
        const safeProbability = Math.max(0, Math.min(100, Number(probabilityPct) || 0));
        if (safeProbability > 60) {
            return { label: "High Risk", color: "#ef4444" };
        }
        if (safeProbability > 30) {
            return { label: "Moderate Risk", color: "#f59e0b" };
        }
        return { label: "Low Risk", color: "#22c55e" };
    }

    function renderGauge(probabilityPct) {
        const canvas = document.getElementById("riskGaugeChart");
        const safeProbability = Math.max(0, Math.min(100, Number(probabilityPct) || 0));
        const gaugeMeta = getRiskGaugeConfig(safeProbability);
        const riskValue = document.getElementById("riskValue");
        const riskLabel = document.getElementById("riskLabel");

        if (riskValue) {
            riskValue.textContent = `${safeProbability.toFixed(1)}%`;
            riskValue.style.color = gaugeMeta.color;
        }
        if (riskLabel) {
            riskLabel.textContent = gaugeMeta.label;
            riskLabel.style.color = gaugeMeta.color;
        }

        if (!canvas) {
            return;
        }
        if (gaugeChart) {
            gaugeChart.destroy();
        }
        gaugeChart = new Chart(canvas, {
            type: "doughnut",
            data: {
                labels: ["Predicted Risk", "Remaining"],
                datasets: [
                    {
                        data: [safeProbability, Math.max(0, 100 - safeProbability)],
                        backgroundColor: [gaugeMeta.color, "#e5e7eb"],
                        borderWidth: 0,
                        hoverOffset: 0,
                    },
                ],
            },
            options: {
                circumference: 180,
                rotation: 270,
                cutout: "72%",
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 1200, easing: "easeOutQuart" },
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
            },
        });
    }

    function renderPredictionCard(result) {
        const summary = document.getElementById("predictionSummary");
        const card = document.getElementById("predictionResultCard");
        const riskStrip = document.getElementById("riskStrip");
        const contributors = document.getElementById("contributorsList");
        const aiExplanationList = document.getElementById("aiExplanationList");
        const modeBadge = document.getElementById("resultModeBadge");
        const explanation = document.getElementById("resultExplanation");

        if (!result) {
            return;
        }

        if (!riskStrip) {
            renderGauge(result.probability_pct || 0);
            return;
        }

        if (summary) {
            summary.classList.add("d-none");
        }
        if (card) {
            card.classList.remove("d-none");
        }

        if (modeBadge) {
            modeBadge.textContent = result.mode_label || "Basic Screening";
        }
        if (explanation) {
            explanation.textContent = result.explanation || "";
        }

        riskStrip.className = `risk-strip risk-${String(result.risk || "").toLowerCase()} mb-3`;
        riskStrip.innerHTML = `
            <div>
                <span class="small text-uppercase">Risk tier</span>
                <h4 class="mb-0">${result.risk}</h4>
            </div>
            <div class="text-end">
                <span class="small text-uppercase">Probability</span>
                <h4 class="mb-0">${result.probability_pct}%</h4>
            </div>
        `;

        if (contributors) {
            contributors.innerHTML = (result.contributors || [])
                .map((item) => `<span class="chip-soft">${item}</span>`)
                .join("");
        }

        if (aiExplanationList) {
            const explanationEntries = Object.entries(result.ai_explanation || {});
            aiExplanationList.innerHTML = explanationEntries.length
                ? explanationEntries
                    .map(([feature, score]) => `
                        <div class="list-group-item px-0 d-flex justify-content-between align-items-center">
                            <span>${feature}</span>
                            <strong>${score}</strong>
                        </div>
                    `)
                    .join("")
                : `<div class="list-group-item px-0 text-muted">Per-feature explanation will appear after a SHAP-enabled prediction.</div>`;
        }

        renderGauge(result.probability_pct || 0);
    }

    function clearFieldErrors(stepElement = null) {
        const scope = stepElement || document;
        scope.querySelectorAll("[data-field-error]").forEach((node) => {
            node.textContent = "";
        });
    }

    function renderFieldErrors(errors = {}) {
        Object.entries(errors).forEach(([fieldName, message]) => {
            const target = document.querySelector(`[data-field-error="${fieldName}"]`);
            if (target) {
                target.textContent = message;
            }
        });
    }

    function getActiveSteps() {
        const clinicalMode = modeInput && modeInput.value === "clinical";
        return steps.filter((step) => clinicalMode || !step.dataset.clinicalStep);
    }

    function getActiveIndicators() {
        const clinicalMode = modeInput && modeInput.value === "clinical";
        return indicators.filter((indicator) => clinicalMode || !indicator.classList.contains("clinical-step-indicator"));
    }

    function updateStepUI() {
        const activeSteps = getActiveSteps();
        const activeIndicators = getActiveIndicators();
        if (!activeSteps.length) {
            return;
        }

        if (currentStepIndex >= activeSteps.length) {
            currentStepIndex = activeSteps.length - 1;
        }

        steps.forEach((step) => step.classList.remove("active"));
        activeSteps[currentStepIndex].classList.add("active");

        indicators.forEach((indicator) => {
            indicator.classList.add("hidden");
            indicator.classList.remove("active");
        });
        activeIndicators.forEach((indicator, index) => {
            indicator.classList.remove("hidden");
            if (index === currentStepIndex) {
                indicator.classList.add("active");
            }
            const badge = indicator.querySelector(".step-badge");
            if (badge) {
                badge.textContent = index + 1;
            }
        });

        const totalSteps = activeSteps.length;
        const currentStep = currentStepIndex + 1;
        const completion = Math.round((currentStep / totalSteps) * 100);

        if (progressBar) {
            progressBar.style.width = `${completion}%`;
        }
        if (progressLabel) {
            progressLabel.textContent = `Step ${currentStep} of ${totalSteps}`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${completion}%`;
        }

        if (prevBtn) {
            prevBtn.disabled = currentStepIndex === 0;
        }
        if (nextBtn) {
            nextBtn.classList.toggle("d-none", currentStepIndex === totalSteps - 1);
        }
        if (submitBtn) {
            submitBtn.classList.toggle("d-none", currentStepIndex !== totalSteps - 1);
        }
    }

    function updateClinicalVisibility() {
        const clinicalMode = modeInput && modeInput.value === "clinical";
        document.querySelectorAll("[data-clinical-field='true']").forEach((field) => {
            field.disabled = !clinicalMode;
            if (!clinicalMode && field.tagName === "SELECT") {
                field.value = "";
            }
            if (!clinicalMode && field.tagName === "INPUT") {
                field.value = "";
            }
        });
        document.querySelectorAll(".clinical-step").forEach((step) => {
            step.classList.toggle("d-none", !clinicalMode);
        });
        updateStepUI();
    }

    function validateCurrentStep() {
        const activeSteps = getActiveSteps();
        const activeStep = activeSteps[currentStepIndex];
        if (!activeStep) {
            return true;
        }

        clearFieldErrors(activeStep);
        let isValid = true;
        activeStep.querySelectorAll("input[name], select[name]").forEach((field) => {
            if (field.disabled || field.type === "hidden") {
                return;
            }
            const errorTarget = activeStep.querySelector(`[data-field-error="${field.name}"]`);
            const value = String(field.value || "").trim();
            if (!value) {
                isValid = false;
                if (errorTarget) {
                    errorTarget.textContent = "This field is required.";
                }
                return;
            }

            if (field.type === "number") {
                const numericValue = Number(value);
                const min = field.getAttribute("min");
                const max = field.getAttribute("max");
                if (Number.isNaN(numericValue)) {
                    isValid = false;
                    if (errorTarget) {
                        errorTarget.textContent = "Enter a numeric value.";
                    }
                    return;
                }
                if (min !== null && numericValue < Number(min)) {
                    isValid = false;
                    if (errorTarget) {
                        errorTarget.textContent = `Enter a value greater than or equal to ${min}.`;
                    }
                }
                if (max !== null && numericValue > Number(max)) {
                    isValid = false;
                    if (errorTarget) {
                        errorTarget.textContent = `Enter a value less than or equal to ${max}.`;
                    }
                }
            }
        });
        return isValid;
    }

    function updateBmiPreview() {
        if (!bmiValue || !heightInput || !weightInput) {
            return;
        }
        const height = Number(heightInput.value);
        const weight = Number(weightInput.value);
        if (!height || !weight) {
            bmiValue.textContent = "-";
            return;
        }
        const bmi = weight / ((height / 100) ** 2);
        bmiValue.textContent = bmi.toFixed(1);
    }

    function collectFormData() {
        const data = {};
        new FormData(form).forEach((value, key) => {
            data[key] = value;
        });
        return data;
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            if (!validateCurrentStep()) {
                return;
            }
            currentStepIndex += 1;
            updateStepUI();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            currentStepIndex = Math.max(0, currentStepIndex - 1);
            updateStepUI();
        });
    }

    modeRadios.forEach((radio) => {
        radio.addEventListener("change", () => {
            if (!radio.checked || !modeInput) {
                return;
            }
            modeInput.value = radio.value;
            currentStepIndex = 0;
            updateClinicalVisibility();
        });
    });

    [heightInput, weightInput].forEach((input) => {
        if (input) {
            input.addEventListener("input", updateBmiPreview);
        }
    });

    if (form) {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            clearFieldErrors();
            app.showMessage(messageTarget, "info", "");
            if (!validateCurrentStep()) {
                return;
            }
            if (currentStepIndex !== getActiveSteps().length - 1) {
                currentStepIndex += 1;
                updateStepUI();
                return;
            }
            try {
                const payload = collectFormData();
                const response = await app.fetchJSON(form.dataset.endpoint, {
                    method: "POST",
                    body: JSON.stringify(payload),
                });
                renderPredictionCard(response.result);
                app.showMessage(messageTarget, "success", "Prediction completed. The analysis dashboard now reflects this submission.");
                window.dispatchEvent(new CustomEvent("prediction:updated", { detail: response.analysis }));
            } catch (error) {
                renderFieldErrors(error.payload?.errors || {});
                app.showMessage(messageTarget, "danger", error.message || "Prediction failed.");
            }
        });
    }

    const initialPrediction = app.readJSONScript("initial-prediction-data");
    if (initialPrediction && initialPrediction.result) {
        renderPredictionCard(initialPrediction.result);
    }

    updateBmiPreview();
    updateClinicalVisibility();
})();
