(() => {
    const app = window.PCOSApp;
    const form = document.getElementById("videoSearchForm");
    const results = document.getElementById("videoResults");
    const message = document.getElementById("videoMessage");
    const modalEl = document.getElementById("videoPlayerModal");
    const modalTitle = document.getElementById("videoPlayerTitle");
    const modalFrame = document.getElementById("videoPlayerFrame");
    if (!app || !form || !results) {
        return;
    }

    const playerModal = modalEl ? new bootstrap.Modal(modalEl) : null;

    function renderVideos(items) {
        results.innerHTML = items
            .map((item) => `
                <div class="col-lg-4 col-md-6">
                    <div class="video-card">
                        <button
                            type="button"
                            class="video-thumbnail-button"
                            data-title="${item.title}"
                            data-embed-url="${item.embed_url}"
                        >
                            <img class="video-thumbnail" src="${item.thumbnail}" alt="${item.title}">
                            <span class="video-play-button"><i class="fa-solid fa-play"></i></span>
                        </button>
                        <div class="video-card-body">
                            <h5 class="mb-2">${item.title}</h5>
                            <p class="video-meta mb-3">${item.channel}</p>
                            <div class="d-flex gap-2">
                                <button
                                    type="button"
                                    class="btn btn-primary flex-grow-1 video-open-button"
                                    data-title="${item.title}"
                                    data-embed-url="${item.embed_url}"
                                >
                                    <i class="fa-solid fa-circle-play me-2"></i>Play
                                </button>
                                <a class="btn btn-outline-primary" href="${item.url}" target="_blank" rel="noopener noreferrer">
                                    <i class="fa-solid fa-arrow-up-right-from-square"></i>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            `)
            .join("");
    }

    function openVideo(title, embedUrl) {
        if (!playerModal || !modalFrame || !modalTitle) {
            return;
        }
        modalTitle.textContent = title;
        modalFrame.src = embedUrl;
        playerModal.show();
    }

    async function loadVideos(query) {
        try {
            const endpoint = `${form.dataset.endpoint}?q=${encodeURIComponent(query)}`;
            const response = await app.fetchJSON(endpoint);
            renderVideos(response.results || []);
            app.showMessage(
                message,
                response.results.length ? (response.message ? "info" : "success") : "warning",
                response.message || `Loaded ${response.results.length} PCOS-related videos.`,
            );
        } catch (error) {
            results.innerHTML = "";
            app.showMessage(message, "danger", error.message || "Video search failed.");
        }
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const query = String(new FormData(form).get("q") || "").trim();
        if (!query) {
            app.showMessage(message, "warning", "Enter a PCOS-related topic to search videos.");
            return;
        }
        loadVideos(query);
    });

    document.querySelectorAll(".video-topic-chip").forEach((button) => {
        button.addEventListener("click", () => {
            const input = form.querySelector('input[name="q"]');
            if (input) {
                input.value = button.dataset.topic || "";
                loadVideos(input.value);
            }
        });
    });

    results.addEventListener("click", (event) => {
        const button = event.target.closest(".video-thumbnail-button, .video-open-button");
        if (!button) {
            return;
        }
        openVideo(button.dataset.title || "PCOS Education Video", button.dataset.embedUrl || "");
    });

    if (modalEl && modalFrame) {
        modalEl.addEventListener("hidden.bs.modal", () => {
            modalFrame.src = "";
        });
    }

    const initialQuery = form.querySelector('input[name="q"]');
    if (initialQuery && initialQuery.value) {
        loadVideos(initialQuery.value);
    }
})();
