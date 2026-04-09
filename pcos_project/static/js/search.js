(() => {
    const app = window.PCOSApp;
    const input = document.getElementById("assistantQueryInput");
    const suggestionBox = document.getElementById("suggestionBox");
    const dropdown = document.getElementById("searchSuggestionDropdown");
    const faqMatchList = document.getElementById("faqMatchList");
    const message = document.getElementById("searchMessage");
    if (!app || !input || !suggestionBox || !dropdown) {
        return;
    }

    function hideDropdown() {
        dropdown.classList.add("d-none");
        dropdown.innerHTML = "";
    }

    function renderSuggestions(suggestions) {
        suggestionBox.innerHTML = suggestions
            .map((item) => `
                <a class="suggestion-chip" href="https://www.google.com/search?q=${encodeURIComponent(item)}" target="_blank" rel="noopener noreferrer">
                    <i class="fa-solid fa-arrow-up-right-from-square"></i>${item}
                </a>
            `)
            .join("");
    }

    function renderDropdown(suggestions) {
        if (!suggestions.length) {
            hideDropdown();
            return;
        }
        dropdown.innerHTML = suggestions
            .map((item) => `
                <button type="button" class="autocomplete-item" data-value="${item}">
                    <i class="fa-solid fa-magnifying-glass text-primary"></i>
                    <span>${item}</span>
                </button>
            `)
            .join("");
        dropdown.classList.remove("d-none");
    }

    function renderFaqMatches(faqs) {
        if (!faqMatchList) {
            return;
        }
        faqMatchList.innerHTML = faqs
            .map((faq) => `
                <div class="article-item">
                    <h6 class="mb-1">${faq.question}</h6>
                    <p class="small mb-0">${faq.answer}</p>
                </div>
            `)
            .join("");
    }

    const loadSuggestions = app.debounce(async () => {
        const query = input.value.trim();
        if (query.length < 2) {
            hideDropdown();
            suggestionBox.innerHTML = "";
            if (faqMatchList) {
                faqMatchList.innerHTML = "";
            }
            app.showMessage(message, "info", "");
            return;
        }
        try {
            const endpoint = `${input.dataset.endpoint}?q=${encodeURIComponent(query)}`;
            const response = await app.fetchJSON(endpoint);
            renderSuggestions(response.suggestions || []);
            renderDropdown(response.suggestions || []);
            renderFaqMatches(response.faqs || []);
            app.showMessage(
                message,
                response.suggestions.length ? "success" : "warning",
                response.message || `Loaded ${response.suggestions.length} PCOS-related suggestion phrases.`,
            );
        } catch (error) {
            hideDropdown();
            suggestionBox.innerHTML = "";
            renderFaqMatches([]);
            app.showMessage(message, "danger", error.message || "Suggestion search failed.");
        }
    }, 250);

    input.addEventListener("input", loadSuggestions);
    input.addEventListener("focus", loadSuggestions);

    dropdown.addEventListener("click", (event) => {
        const button = event.target.closest(".autocomplete-item");
        if (!button) {
            return;
        }
        const value = button.dataset.value || "";
        input.value = value;
        hideDropdown();
        window.open(`https://www.google.com/search?q=${encodeURIComponent(value)}`, "_blank", "noopener,noreferrer");
    });

    document.addEventListener("click", (event) => {
        if (!dropdown.contains(event.target) && event.target !== input) {
            hideDropdown();
        }
    });

    if (input.value.trim().length >= 2) {
        loadSuggestions();
    }
})();
