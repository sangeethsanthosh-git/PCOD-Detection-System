(() => {
    const loader = document.getElementById("globalLoader");

    function getCsrfToken() {
        const tokenEl = document.querySelector('meta[name="csrf-token"]');
        return tokenEl ? tokenEl.getAttribute("content") : "";
    }

    function showLoader() {
        if (loader) {
            loader.classList.add("show");
        }
    }

    function hideLoader() {
        if (loader) {
            loader.classList.remove("show");
        }
    }

    function showMessage(target, type, message) {
        if (!target) {
            return;
        }
        if (!message) {
            target.innerHTML = "";
            return;
        }
        target.innerHTML = `
            <div class="status-banner">
                <div class="alert alert-${type}" role="alert">${message}</div>
            </div>
        `;
    }

    async function fetchJSON(url, options = {}) {
        const config = { ...options };
        const headers = new Headers(config.headers || {});
        headers.set("Accept", "application/json");
        if (config.body && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }
        if (config.method && config.method.toUpperCase() !== "GET") {
            headers.set("X-CSRFToken", getCsrfToken());
        }
        config.headers = headers;

        showLoader();
        try {
            const response = await fetch(url, config);
            const data = await response.json().catch(() => ({}));
            if (!response.ok || data.ok === false) {
                const error = new Error(data.message || "Request failed.");
                error.payload = data;
                throw error;
            }
            return data;
        } finally {
            hideLoader();
        }
    }

    function debounce(callback, delay = 300) {
        let timer;
        return (...args) => {
            window.clearTimeout(timer);
            timer = window.setTimeout(() => callback(...args), delay);
        };
    }

    function readJSONScript(id) {
        const el = document.getElementById(id);
        if (!el) {
            return null;
        }
        try {
            return JSON.parse(el.textContent);
        } catch (error) {
            return null;
        }
    }

    window.PCOSApp = {
        fetchJSON,
        showMessage,
        debounce,
        readJSONScript,
    };
})();
