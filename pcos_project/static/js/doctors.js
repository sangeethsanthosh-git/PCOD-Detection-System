(() => {
    const app = window.PCOSApp;
    const form = document.getElementById("doctorSearchForm");
    const results = document.getElementById("doctorResults");
    const message = document.getElementById("doctorMessage");
    const input = document.getElementById("doctorLocationInput");
    const autocompleteList = document.getElementById("doctorAutocompleteList");
    const currentLocationButton = document.getElementById("useCurrentLocationBtn");
    if (!app || !form || !results || !input || !autocompleteList) {
        return;
    }

    let googleAutocompleteService = null;

    if (window.google && window.google.maps && window.google.maps.places) {
        googleAutocompleteService = new window.google.maps.places.AutocompleteService();
    }

    function hideAutocomplete() {
        autocompleteList.classList.add("d-none");
        autocompleteList.innerHTML = "";
    }

    function renderAutocomplete(items) {
        if (!items.length) {
            hideAutocomplete();
            return;
        }
        autocompleteList.innerHTML = items
            .map((item) => `
                <button type="button" class="autocomplete-item" data-description="${item.description}">
                    <i class="fa-solid fa-location-dot text-primary"></i>
                    <span>${item.description}</span>
                </button>
            `)
            .join("");
        autocompleteList.classList.remove("d-none");
    }

    function renderDoctors(items) {
        results.innerHTML = items
            .map((item) => `
                <div class="col-lg-4 col-md-6">
                    <div class="doctor-card">
                        <div class="doctor-card-body">
                            <div class="d-flex justify-content-between align-items-start gap-3 mb-3">
                                <div>
                                    <h5 class="mb-1">${item.name}</h5>
                                    <p class="doctor-meta mb-1">${item.speciality}</p>
                                    <p class="doctor-meta mb-0">${item.hospital_name || "Gynecology clinic"}</p>
                                </div>
                                <span class="chip-soft"><i class="fa-solid fa-star me-1"></i>${item.rating ?? "N/A"}</span>
                            </div>
                            <div class="doctor-detail-grid mb-3">
                                <div><i class="fa-solid fa-location-dot text-primary me-2"></i>${item.address}</div>
                                <div><i class="fa-solid fa-route text-primary me-2"></i>${item.distance_km ?? "N/A"} km away</div>
                            </div>
                            <a class="btn btn-outline-primary w-100" href="${item.maps_link}" target="_blank" rel="noopener noreferrer">
                                <i class="fa-solid fa-map-location-dot me-2"></i>Open in Google Maps
                            </a>
                        </div>
                    </div>
                </div>
            `)
            .join("");
    }

    async function search(params) {
        try {
            const endpoint = `${form.dataset.endpoint}?${new URLSearchParams(params).toString()}`;
            const response = await app.fetchJSON(endpoint);
            renderDoctors(response.results || []);
            app.showMessage(
                message,
                response.results.length ? "success" : "warning",
                response.message || `Showing ${response.results.length} gynecology-focused hospitals or clinics.`,
            );
        } catch (error) {
            results.innerHTML = "";
            app.showMessage(message, "danger", error.message || "Doctor search failed.");
        }
    }

    async function fetchAutocomplete(query) {
        if (googleAutocompleteService) {
            googleAutocompleteService.getPlacePredictions(
                {
                    input: query,
                    types: ["(cities)"],
                    componentRestrictions: { country: "in" },
                },
                (predictions) => {
                    const items = (predictions || []).slice(0, 6).map((item) => ({
                        description: item.description,
                        place_id: item.place_id,
                    }));
                    renderAutocomplete(items);
                },
            );
            return;
        }

        try {
            const endpoint = `${form.dataset.autocompleteEndpoint}?q=${encodeURIComponent(query)}`;
            const response = await app.fetchJSON(endpoint);
            renderAutocomplete(response.results || []);
        } catch (error) {
            hideAutocomplete();
        }
    }

    const debouncedAutocomplete = app.debounce(() => {
        const query = input.value.trim();
        if (query.length < 2) {
            hideAutocomplete();
            return;
        }
        fetchAutocomplete(query);
    }, 250);

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        hideAutocomplete();
        const location = String(new FormData(form).get("location") || "").trim();
        if (!location) {
            app.showMessage(message, "warning", "Enter a city or use current location to search nearby gynecology care.");
            return;
        }
        search({ location });
    });

    input.addEventListener("input", debouncedAutocomplete);
    input.addEventListener("focus", debouncedAutocomplete);

    autocompleteList.addEventListener("click", (event) => {
        const button = event.target.closest(".autocomplete-item");
        if (!button) {
            return;
        }
        input.value = button.dataset.description || "";
        hideAutocomplete();
    });

    document.addEventListener("click", (event) => {
        if (!autocompleteList.contains(event.target) && event.target !== input) {
            hideAutocomplete();
        }
    });

    if (currentLocationButton) {
        currentLocationButton.addEventListener("click", () => {
            if (!navigator.geolocation) {
                app.showMessage(message, "warning", "Browser geolocation is not available on this device.");
                return;
            }
            app.showMessage(message, "info", "Detecting your current location...");
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    search({
                        lat: position.coords.latitude.toFixed(6),
                        lon: position.coords.longitude.toFixed(6),
                    });
                },
                (error) => {
                    const messageMap = {
                        1: "Location permission was denied. Enter your city manually to continue.",
                        2: "Location could not be determined. Try again or enter your city manually.",
                        3: "Location request timed out. Try again or use manual search.",
                    };
                    app.showMessage(message, "warning", messageMap[error.code] || "Location could not be determined.");
                },
                {
                    enableHighAccuracy: true,
                    timeout: 12000,
                },
            );
        });
    }

    document.querySelectorAll(".doctor-location-chip").forEach((button) => {
        button.addEventListener("click", () => {
            input.value = button.dataset.location || "";
            hideAutocomplete();
            search({ location: input.value });
        });
    });
})();
