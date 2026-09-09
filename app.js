/**
 * GEO-MN // Enterprise Manganese Geological & Production Intelligence
 * Client-Side Controller & Telemetry Engine
 */

(function () {
    'use strict';

    // ── Configuration & State ────────────────────────────────
    // Automatically detects whether running locally or deployed on the cloud
    const API_BASE = window.location.origin;

    const state = {
        currentSection: 'dashboard',
        apiOnline: false,
        reserveGridData: null,
        reserveMap: null,
        heatLayer: null,
        selectedCircle: null,
        selectedMarker: null,
        baseLayers: {},
        currentBaseLayer: null,
        simHistory: [],
    };

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // ── Toast System ─────────────────────────────────────────
    function showToast(message, type = 'info') {
        const container = $('#toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(30px)';
            setTimeout(() => toast.remove(), 250);
        }, 4000);
    }

    function setLoading(btn, loading) {
        if (!btn) return;
        if (loading) {
            btn.classList.add('loading');
            btn.disabled = true;
        } else {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    }

    // ── HTTP API Client ──────────────────────────────────────
    async function apiGet(endpoint) {
        const res = await fetch(`${API_BASE}${endpoint}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return res.json();
    }

    // FastAPI validation errors (422) arrive as detail: [{loc, msg, type}, ...].
    // Flatten them into something readable instead of "[object Object]".
    function describeApiError(err, status) {
        const d = err && err.detail;
        if (Array.isArray(d)) {
            return d.map(e => {
                const field = Array.isArray(e.loc) ? e.loc.filter(x => x !== 'body').join('.') : '?';
                return `${field}: ${e.msg}`;
            }).join('; ');
        }
        if (typeof d === 'string') return d;
        return `HTTP ${status}`;
    }

    async function apiPost(endpoint, body) {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            const msg = describeApiError(err, res.status);
            console.error(`POST ${endpoint} -> ${res.status}`, { sent: body, response: err });
            throw new Error(msg);
        }
        return res.json();
    }

    // ── Navigation Controller ────────────────────────────────
    const sectionTitles = {
        dashboard: 'Executive Overview',
        reserve: 'Satellite Prospecting & Geological Mapping',
        shortfall: 'Production Shortfall Risk Engine',
        simulator: 'What-If Scenario Simulation Ledger',
    };

    function initNavigation() {
        $$('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const sec = link.dataset.section;
                showSection(sec);
                $('#sidebar')?.classList.remove('open');
            });
        });

        $$('.action-btn[data-navigate]').forEach(btn => {
            btn.addEventListener('click', () => showSection(btn.dataset.navigate));
        });

        $('#mobileToggle')?.addEventListener('click', () => {
            $('#sidebar')?.classList.toggle('open');
        });
    }

    const VALID_SECTIONS = ['dashboard', 'reserve', 'shortfall', 'simulator'];

    function showSection(sectionId, updateHash = true) {
        if (!VALID_SECTIONS.includes(sectionId)) sectionId = 'dashboard';
        state.currentSection = sectionId;

        // Reflect the section in the URL so a reload (or a shared link) lands
        // back here instead of resetting to the dashboard. replaceState avoids
        // stacking a history entry for every sidebar click.
        if (updateHash && window.location.hash !== `#${sectionId}`) {
            history.replaceState(null, '', `#${sectionId}`);
        }

        $$('.nav-link').forEach(l => l.classList.remove('active'));
        $(`.nav-link[data-section="${sectionId}"]`)?.classList.add('active');

        $$('.section').forEach(s => s.classList.remove('active'));
        const targetSec = $(`#section-${sectionId}`);
        if (targetSec) {
            targetSec.classList.add('active');
        }

        const title = sectionTitles[sectionId];
        if (title && $('#pageTitle')) {
            $('#pageTitle').textContent = title;
        }

        if (sectionId === 'reserve' && state.reserveMap) {
            setTimeout(() => {
                state.reserveMap.invalidateSize();
            }, 120);
        }
    }

    // ── Module 1: Dashboard Telemetry ────────────────────────
    async function initDashboard() {
        try {
            const data = await apiGet('/');
            state.apiOnline = data.status === 'ok';

            if ($('#statApiStatus')) {
                $('#statApiStatus').textContent = data.status === 'ok' ? 'HEALTHY' : 'DEGRADED';
            }
            if ($('#statReserveGrid')) {
                $('#statReserveGrid').textContent = data.reserve_cache_loaded ? '768 CELLS' : 'OFFLINE';
            }
            if ($('#statProdModel')) {
                $('#statProdModel').textContent = data.production_model_loaded ? 'ONLINE' : 'UNLOADED';
            }
            if ($('#statShap')) {
                $('#statShap').textContent = data.shap_available ? 'SHAP ACTIVE' : 'HEURISTIC';
            }

            $('#statusDot')?.classList.toggle('online', data.status === 'ok');
            if ($('#statusText')) {
                $('#statusText').textContent = data.status === 'ok' ? 'TELEMETRY ONLINE' : 'SYSTEM OFFLINE';
            }
        } catch (e) {
            if ($('#statApiStatus')) $('#statApiStatus').textContent = 'UNREACHABLE';
            if ($('#statusText')) $('#statusText').textContent = 'SERVER DISCONNECTED';
            showToast('Telemetry gateway unreachable at ' + API_BASE, 'error');
        }
    }

    // ── Module 2: Satellite Prospecting GIS ──────────────────
    async function initReserveMap() {
        if (!window.L) {
            console.error('Leaflet GIS library not loaded');
            return;
        }

        const map = L.map('reserveMap', {
            center: [21.81, 80.23],
            zoom: 8,
            minZoom: 5,
            maxZoom: 16,
            zoomControl: true,
        });
        state.reserveMap = map;

        // Base Tile Layers
        const satTile = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Esri, USGS, AeroGRID, IGN',
            maxZoom: 18,
        });

        const satLabels = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
            attribution: 'CARTO',
            maxZoom: 18,
            subdomains: 'abcd',
        });

        const nasaLSTTile = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_Land_Surface_Temp_Day/default/2024-05-01/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png', {
            attribution: 'NASA GIBS &mdash; MODIS Land Surface Temp',
            maxNativeZoom: 7,
            maxZoom: 18,
            opacity: 0.9,
        });

        const streetTile = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: 'OpenStreetMap',
            maxZoom: 18,
        });

        // Layer Groups
        const satGroup = L.layerGroup([satTile, satLabels]).addTo(map);
        const nasaThermalGroup = L.layerGroup([satTile, nasaLSTTile, satLabels]);

        state.baseLayers = {
            satellite: satGroup,
            thermal: nasaThermalGroup,
            street: streetTile,
        };
        state.currentBaseLayer = satGroup;

        // Build Heatmap Layer (kept off by default for clean pure satellite)
        try {
            state.reserveGridData = await apiGet('/reserve_grid');
            renderLeafletHeatmap(map, state.reserveGridData);
        } catch (e) {
            console.warn('Reserve grid fetch error:', e);
        }

        // Segmented Layer Switchers
        setupLayerSwitchers(map);

        // Real-time Cursor Coordinate Readout
        map.on('mousemove', (e) => {
            const readout = $('#cursorCoordReadout');
            if (readout) {
                readout.textContent = `Cursor: ${e.latlng.lat.toFixed(4)}°N, ${e.latlng.lng.toFixed(4)}°E`;
            }
        });

        // Map Click Target Listener
        map.on('click', async (e) => {
            const lat = parseFloat(e.latlng.lat.toFixed(4));
            const lon = parseFloat(e.latlng.lng.toFixed(4));
            $('#reserveLat').value = lat.toFixed(4);
            $('#reserveLon').value = lon.toFixed(4);
            await predictReserve(lat, lon, false);
        });

        // Manual Input Target Submit
        $('#btnPredictReserve')?.addEventListener('click', async (e) => {
            e.preventDefault();
            const lat = parseFloat($('#reserveLat').value);
            const lon = parseFloat($('#reserveLon').value);
            if (isNaN(lat) || isNaN(lon)) {
                showToast('Please enter valid latitude and longitude coordinates', 'error');
                return;
            }
            await predictReserve(lat, lon, true);
        });

        // Sector Preset Quick Chips
        $$('.geo-preset-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const lat = parseFloat(chip.dataset.lat);
                const lon = parseFloat(chip.dataset.lon);
                $('#reserveLat').value = lat.toFixed(4);
                $('#reserveLon').value = lon.toFixed(4);
                predictReserve(lat, lon, true);
            });
        });

        // Heatmap Opacity Slider
        $('#heatOpacityRange')?.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if ($('#heatOpacityVal')) $('#heatOpacityVal').textContent = Math.round(val * 100) + '%';
            updateHeatmapOpacity(val);
        });

        // Heatmap Overlay Toggle
        $('#toggleHeatmapOverlay')?.addEventListener('change', (e) => {
            setHeatmapVisible(e.target.checked);
            if (e.target.checked) {
                updateLegend('deposit');
            } else if (state.currentBaseLayer === state.baseLayers.thermal) {
                updateLegend('thermal');
            } else {
                updateLegend('satellite');
            }
        });

        // Reset Center Button
        $('#btnResetMap')?.addEventListener('click', () => {
            map.flyTo([21.81, 80.23], 8, { duration: 1 });
        });

        // Initial Prediction at Balaghat Pit
        setTimeout(() => {
            predictReserve(21.81, 80.23, false);
        }, 400);
    }

    function setupLayerSwitchers(map) {
        const btns = {
            satellite: $('#btnLayerSatellite'),
            thermal: $('#btnLayerThermal'),
            street: $('#btnLayerStreet'),
        };

        const heatToggle = $('#toggleHeatmapOverlay');

        Object.keys(btns).forEach(type => {
            const btn = btns[type];
            if (!btn) return;
            btn.addEventListener('click', () => {
                if (state.currentBaseLayer && state.currentBaseLayer !== state.baseLayers[type]) {
                    map.removeLayer(state.currentBaseLayer);
                }
                state.currentBaseLayer = state.baseLayers[type].addTo(map);

                if (type === 'satellite') {
                    if (heatToggle) heatToggle.checked = false;
                    setHeatmapVisible(false);
                    updateLegend('satellite');
                } else if (type === 'thermal') {
                    if (heatToggle) heatToggle.checked = false;
                    setHeatmapVisible(false);
                    updateLegend('thermal');
                } else {
                    if (heatToggle) setHeatmapVisible(heatToggle.checked);
                    updateLegend(heatToggle && heatToggle.checked ? 'deposit' : 'satellite');
                }

                if (state.heatLayer && map.hasLayer(state.heatLayer)) state.heatLayer.bringToFront?.();
                if (state.selectedCircle) state.selectedCircle.bringToFront?.();
                if (state.selectedMarker) state.selectedMarker.bringToFront?.();

                Object.values(btns).forEach(b => b?.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    function setHeatmapVisible(visible) {
        if (!state.reserveMap || !state.heatLayer) return;
        const map = state.reserveMap;
        if (visible) {
            if (!map.hasLayer(state.heatLayer)) map.addLayer(state.heatLayer);
            const opacitySlider = $('#heatOpacityRange');
            if (opacitySlider) updateHeatmapOpacity(parseFloat(opacitySlider.value));
            const group = $('#heatOpacityGroup');
            if (group) group.style.display = 'flex';
        } else {
            if (map.hasLayer(state.heatLayer)) map.removeLayer(state.heatLayer);
            const group = $('#heatOpacityGroup');
            if (group) group.style.display = 'none';
        }
    }

    function updateHeatmapOpacity(val) {
        const canvas = document.querySelector('.leaflet-heatmap-layer');
        if (canvas) {
            canvas.style.opacity = val;
            canvas.style.transition = 'opacity 0.2s ease';
        }
    }

    function renderLeafletHeatmap(map, data) {
        if (!window.L || !window.L.heatLayer || !data || data.length === 0) return;
        // Heatmap intensity uses rank so the layer spreads across the full
        // colour range instead of saturating (raw probabilities cluster 0.6-1.0).
        const heatPoints = data.map(d => [d.lat, d.lon,
            Math.max(0.05, (typeof d.rank === 'number') ? d.rank / 100 : d.probability)]);
        const heat = L.heatLayer(heatPoints, {
            radius: 28,
            blur: 22,
            maxZoom: 11,
            max: 1.0,
            minOpacity: 0.35,
            gradient: {
                0.15: '#3b82f6',
                0.35: '#06b6d4',
                0.55: '#10b981',
                0.75: '#f59e0b',
                0.95: '#ef4444'
            }
        });
        state.heatLayer = heat;
    }

    function updateLegend(mode) {
        const title = $('#legendTitle');
        const sub = $('#legendSubtitle');
        const bar = $('#legendBar');
        const labels = $('#legendLabels');
        const hint = $('#legendHint');
        const hud = document.querySelector('.floating-legend-hud');
        if (!title) return;

        // In plain satellite mode there is no data scale to explain — the old
        // "Dense Foliage / Exposed Rock" bar was decorative, not derived from
        // anything. Hide the whole HUD rather than show a meaningless legend.
        if (hud) hud.style.display = (mode === 'satellite') ? 'none' : '';

        if (mode === 'thermal') {
            title.textContent = 'Thermal Map from Satellite (NASA MODIS LST)';
            if (sub) sub.textContent = 'Infrared Land Surface Radiation (°C)';
            if (bar) bar.style.background = 'linear-gradient(90deg, #313695 0%, #4575b4 20%, #74add1 40%, #fee090 60%, #f46d43 80%, #a50026 100%)';
            if (labels) labels.innerHTML = '<span>Cool (&lt;25°C)</span><span>Moderate (35°C)</span><span>High Thermal Radiation (&gt;50°C)</span>';
            if (hint) hint.textContent = 'Real-time satellite infrared thermal radiometry. High thermal inertia indicates exposed mineralized rock outcrops and active quarries.';
        } else if (mode === 'satellite') {
            title.textContent = 'High-Resolution Satellite Terrain';
            if (sub) sub.textContent = 'Esri World Imagery + Regional Geographic Labels';
            if (bar) bar.style.background = 'linear-gradient(90deg, #1e293b 0%, #334155 50%, #64748b 100%)';
            if (labels) labels.innerHTML = '<span>Dense Foliage</span><span>Vegetation / Soil</span><span>Exposed Ground / Rock</span>';
            if (hint) hint.textContent = 'Pure satellite terrain view without overlays. Click anywhere on the terrain to inspect coordinates and place a target crosshair.';
        } else {
            title.textContent = 'Relative Manganese Prospectivity Rank';
            if (sub) sub.textContent = 'Percentile within the analyzed belt (VNIR/SWIR/DEM/LST features)';
            if (bar) bar.style.background = 'linear-gradient(90deg, #3b82f6 0%, #06b6d4 25%, #10b981 50%, #f59e0b 75%, #ef4444 100%)';
            if (labels) labels.innerHTML = '<span>Rank 0 — deprioritize</span><span>Rank 50 — moderate</span><span>Rank 100 — top drill target</span>';
            if (hint) hint.textContent = 'Model confidence is derived from hydrothermal clay alteration, iron oxide capping, topography, and thermal signatures.';
        }
    }

    async function predictReserve(lat, lon, fly = true) {
        const btn = $('#btnPredictReserve');
        setLoading(btn, true);
        try {
            const data = await apiPost('/predict_reserve', { lat, lon });
            renderGeologicalTelemetry(data);
            highlightTargetOnMap(lat, lon, data);

            if (fly && state.reserveMap) {
                state.reserveMap.flyTo([lat, lon], Math.max(state.reserveMap.getZoom(), 8), { duration: 1 });
            }
        } catch (e) {
            showToast('Reserve query failed: ' + e.message, 'error');
        } finally {
            setLoading(btn, false);
        }
    }

    function renderGeologicalTelemetry(data) {
        // Display the percentile RANK, not the raw probability. The classifier
        // is not calibrated in absolute terms; its ordering is what we trust.
        const rank = (typeof data.rank === 'number') ? data.rank : data.probability * 100;
        const frac = rank / 100;
        const pct = rank.toFixed(1);

        const ring = $('#gaugeProgressRing');
        if (ring) {
            const circumference = 301.6;
            const offset = circumference * (1 - frac);
            ring.style.strokeDashoffset = offset;
            ring.style.stroke = frac > 0.75 ? 'var(--accent-emerald)' : (frac > 0.5 ? 'var(--risk-medium)' : 'var(--risk-high)');
        }
        if ($('#reserveProbValue')) $('#reserveProbValue').textContent = `${pct}%`;

        const badge = $('#geolClassBadge');
        if (badge) {
            badge.textContent = data.tier || 'RANK UNAVAILABLE';
            const style = frac >= 0.75
                ? ['var(--accent-emerald)', 'rgba(16, 185, 129, 0.1)', 'rgba(16, 185, 129, 0.3)']
                : frac >= 0.50
                    ? ['var(--risk-medium)', 'rgba(245, 158, 11, 0.1)', 'rgba(245, 158, 11, 0.3)']
                    : ['var(--text-muted)', 'rgba(255, 255, 255, 0.03)', 'var(--border-subtle)'];
            badge.style.color = style[0];
            badge.style.background = style[1];
            badge.style.borderColor = style[2];
        }

        if ($('#resQueryLatLon')) {
            $('#resQueryLatLon').textContent = `${data.query_lat.toFixed(4)}°N, ${data.query_lon.toFixed(4)}°E`;
        }

        const isLive = data.source === 'live_satellite';

        if ($('#resGridLatLon')) {
            $('#resGridLatLon').textContent = isLive
                ? 'Direct live extraction — no grid lookup needed'
                : `${data.nearest_grid_lat.toFixed(2)}°N, ${data.nearest_grid_lon.toFixed(2)}°E`;
        }

        if ($('#resDistance')) {
            $('#resDistance').textContent = isLive
                ? '0 km (exact coordinate)'
                : `${data.grid_distance_degrees.toFixed(4)}° (~${(data.grid_distance_degrees * 111).toFixed(1)} km)`;
        }

        const sourceEl = $('#resDataSource');
        if (sourceEl) {
            sourceEl.textContent = isLive ? '🛰️ LIVE SATELLITE EXTRACTION' : '📊 CACHED GRID (nearest analyzed point)';
            sourceEl.style.color = isLive ? 'var(--accent-emerald)' : 'var(--text-muted)';
        }
    }

    function highlightTargetOnMap(lat, lon, data) {
        if (!state.reserveMap) return;
        const map = state.reserveMap;
        if (state.selectedCircle) map.removeLayer(state.selectedCircle);
        if (state.selectedMarker) map.removeLayer(state.selectedMarker);
        const r = (data && typeof data.rank === 'number') ? data.rank : 50;
        const color = r > 75 ? '#10b981' : (r > 50 ? '#f59e0b' : '#ef4444');

        state.selectedCircle = L.circle([lat, lon], {
            radius: 16000, color: color, fillColor: color, fillOpacity: 0.16, weight: 2, dashArray: '4, 4'
        }).addTo(map);

        state.selectedMarker = L.circleMarker([lat, lon], {
            radius: 6, color: '#ffffff', fillColor: color, fillOpacity: 1, weight: 2
        }).addTo(map);

        const isLive = data && data.source === 'live_satellite';
        const offsetLine = isLive
            ? 'Live satellite extraction (exact point)'
            : (data && typeof data.grid_distance_degrees === 'number'
                ? `Offset: ${(data.grid_distance_degrees * 111).toFixed(1)} km from grid cell`
                : '');

        const prob = (data && typeof data.probability === 'number') ? data.probability : 0.5;

        const popupHtml = `
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 4px;">
                <div style="font-weight: 700; color: ${color}; margin-bottom: 4px;">TARGET COORDINATES</div>
                <div>Lat: ${lat.toFixed(4)}°N</div>
                <div>Lon: ${lon.toFixed(4)}°E</div>
                <div style="margin-top: 4px; font-weight: 700; color: #fff;">Deposit Probability: ${(prob * 100).toFixed(1)}%</div>
                <div style="color: #94a3b8; font-size: 10px;">${offsetLine}</div>
            </div>
        `;
        state.selectedMarker.bindPopup(popupHtml).openPopup();
    }

    // ── Module 3: Production Shortfall Engine ────────────────
    function initShortfallEngine() {
        // Slider Live Text Bindings
        setupRangeSlider('sf-equipment_availability', 'val-equip-avail', v => `${v} (${Math.round(v * 100)}%)`);
        setupRangeSlider('sf-drilling_delay', 'val-drill-delay', v => `${v} hrs`);
        setupRangeSlider('sf-rainfall', 'val-rainfall', v => `${v} mm`);
        setupRangeSlider('sf-soil_moisture', 'val-soil-moisture', v => `${v}`);

        // Operational Preset Buttons
        $('#presetOptimal')?.addEventListener('click', () => {
            setFormData('sf', {
                equipment_availability: 0.92,
                equipment_downtime: 1.0,
                maintenance_hours: 2.0,
                truck_count: 18,
                rainfall: 2.0,
                soil_moisture: 0.15,
                temperature: 30.0,
                drilling_delay: 0.5,
                blast_delay: 0.4,
                haulage_delay: 0.5,
                target_production: 1000
            });
            showToast('Loaded Preset: Normal Dry Shift (Optimal)');
        });

        $('#presetMonsoon')?.addEventListener('click', () => {
            setFormData('sf', {
                equipment_availability: 0.76,
                equipment_downtime: 4.5,
                maintenance_hours: 5.0,
                truck_count: 12,
                rainfall: 65.0,
                soil_moisture: 0.85,
                temperature: 26.0,
                drilling_delay: 3.0,
                blast_delay: 2.5,
                haulage_delay: 3.5,
                target_production: 1000
            });
            showToast('Loaded Preset: Heavy Monsoon Downpour (Severe Risk)');
        });

        $('#presetCrisis')?.addEventListener('click', () => {
            setFormData('sf', {
                equipment_availability: 0.52,
                equipment_downtime: 8.0,
                maintenance_hours: 10.0,
                truck_count: 8,
                rainfall: 15.0,
                soil_moisture: 0.30,
                temperature: 34.0,
                drilling_delay: 4.0,
                blast_delay: 1.5,
                haulage_delay: 4.5,
                target_production: 1000
            });
            showToast('Loaded Preset: Fleet Breakdown Crisis (Critical Risk)');
        });

        // Form Submit
        $('#shortfallForm')?.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = $('#btnPredictShortfall');
            setLoading(btn, true);

            const payload = gatherFormData('sf');
            try {
                const data = await apiPost('/predict_shortfall', payload);
                renderShortfallResults(data);
            } catch (err) {
                showToast('Shortfall calculation failed: ' + err.message, 'error');
            } finally {
                setLoading(btn, false);
            }
        });
    }

    function setupRangeSlider(sliderId, readoutId, formatter) {
        const slider = $(`#${sliderId}`);
        const readout = $(`#${readoutId}`);
        if (!slider || !readout) return;
        slider.addEventListener('input', (e) => {
            readout.textContent = formatter ? formatter(e.target.value) : e.target.value;
        });
    }

    function setFormData(prefix, vals) {
        Object.keys(vals).forEach(key => {
            const input = $(`#${prefix}-${key}`);
            if (input) {
                input.value = vals[key];
                input.dispatchEvent(new Event('input'));
            }
        });
    }

    function gatherFormData(prefix) {
        return {
            equipment_availability: parseFloat($(`#${prefix}-equipment_availability`).value),
            equipment_downtime: parseFloat($(`#${prefix}-equipment_downtime`).value),
            maintenance_hours: parseFloat($(`#${prefix}-maintenance_hours`).value),
            drilling_delay: parseFloat($(`#${prefix}-drilling_delay`).value),
            blast_delay: parseFloat($(`#${prefix}-blast_delay`).value),
            rainfall: parseFloat($(`#${prefix}-rainfall`).value),
            soil_moisture: parseFloat($(`#${prefix}-soil_moisture`).value),
            temperature: parseFloat($(`#${prefix}-temperature`).value),
            truck_count: parseInt($(`#${prefix}-truck_count`).value, 10),
            haulage_delay: parseFloat($(`#${prefix}-haulage_delay`).value),
            target_production: parseFloat($(`#${prefix}-target_production`).value),
        };
    }

    // ── SHAP root-cause horizontal bar chart ─────────────────
    // Bars are scaled to the LARGEST contributor rather than to 100, so the
    // top driver always fills the track and the relative sizes stay readable
    // even when the top cause is only 30% of total attributed loss.
    const CAUSE_LABELS = {
        equipment_availability: 'Equipment availability',
        equipment_downtime: 'Equipment downtime',
        maintenance_hours: 'Maintenance hours',
        drilling_delay: 'Drilling delay',
        blast_delay: 'Blast delay',
        rainfall: 'Rainfall',
        soil_moisture: 'Soil moisture',
        temperature: 'Temperature',
        truck_count: 'Truck count',
        haulage_delay: 'Haulage delay',
    };

    function renderRootCauseChart(container, causes) {
        if (!container) return;
        container.innerHTML = '';

        if (!causes || typeof causes !== 'object') return;

        const entries = Object.entries(causes).filter(([, v]) => typeof v === 'number');

        // Backend returns {Status: "..."} when SHAP is unavailable or found
        // no negative drivers — show that message instead of an empty chart.
        if (entries.length === 0) {
            const msg = document.createElement('div');
            msg.className = 'cause-empty';
            msg.textContent = Object.values(causes)[0] || 'No attribution available.';
            container.appendChild(msg);
            return;
        }

        entries.sort((a, b) => b[1] - a[1]);
        const max = entries[0][1] || 1;

        const chart = document.createElement('div');
        chart.className = 'shap-chart';

        entries.forEach(([feature, pct], i) => {
            const label = CAUSE_LABELS[feature] || feature.replace(/_/g, ' ');
            const width = Math.max(2, (pct / max) * 100);
            const row = document.createElement('div');
            row.className = 'shap-row';
            row.innerHTML = `
                <div class="shap-label" title="${label}">${label}</div>
                <div class="shap-track">
                    <div class="shap-bar ${i === 0 ? 'shap-bar-top' : ''}" style="width:${width}%"></div>
                </div>
                <div class="shap-value mono-val">${pct.toFixed(1)}%</div>
            `;
            chart.appendChild(row);
        });

        const axis = document.createElement('div');
        axis.className = 'shap-axis';
        axis.textContent = 'Share of attributed shortfall (SHAP, negative contributors only)';

        container.appendChild(chart);
        container.appendChild(axis);
    }

    // Returns gatherFormData(prefix), or `fallback` if the form is missing or
    // any field is blank/non-numeric. parseFloat('') is NaN, and JSON.stringify
    // serialises NaN as null — which Pydantic rejects with a 422.
    function gatherFormDataSafe(prefix, fallback) {
        let data;
        try {
            data = gatherFormData(prefix);
        } catch (e) {
            console.warn(`Form "${prefix}" not found; using fallback values.`);
            return fallback;
        }
        const bad = Object.entries(data).filter(([, v]) => typeof v !== 'number' || !isFinite(v));
        if (bad.length) {
            console.warn(`Form "${prefix}" has invalid fields, using fallback:`, bad.map(b => b[0]));
            return fallback;
        }
        return data;
    }

    // Shift presets. Every value below was run through the deployed model —
    // each genuinely lands in the tier its label claims (7.5 / 12.4 / 20.8 /
    // 55.6 % shortfall). Do not tweak them without re-checking the tier.
    const SHIFT_PRESETS = {
        optimal: { equipment_availability: 0.98, equipment_downtime: 0.3, maintenance_hours: 0.5,
                   drilling_delay: 0.1, blast_delay: 0.1, rainfall: 1.0, soil_moisture: 0.15,
                   temperature: 29.0, truck_count: 28, haulage_delay: 0.2, target_production: 1000 },
        normal:  { equipment_availability: 0.98, equipment_downtime: 0.3, maintenance_hours: 0.5,
                   drilling_delay: 0.1, blast_delay: 0.1, rainfall: 1.0, soil_moisture: 0.15,
                   temperature: 29.0, truck_count: 15, haulage_delay: 0.2, target_production: 1000 },
        strained:{ equipment_availability: 0.80, equipment_downtime: 0.3, maintenance_hours: 0.5,
                   drilling_delay: 0.1, blast_delay: 0.1, rainfall: 15.0, soil_moisture: 0.15,
                   temperature: 29.0, truck_count: 16, haulage_delay: 0.2, target_production: 1000 },
        monsoon: { equipment_availability: 0.78, equipment_downtime: 4.5, maintenance_hours: 6.0,
                   drilling_delay: 2.5, blast_delay: 1.5, rainfall: 45.0, soil_moisture: 0.72,
                   temperature: 27.0, truck_count: 12, haulage_delay: 2.0, target_production: 1000 },
    };

    function applyPreset(prefix, preset) {
        Object.entries(preset).forEach(([field, value]) => {
            const el = $(`#${prefix}-${field}`);
            if (!el) return;
            el.value = value;
            // Range inputs need an input event so their value pill updates.
            el.dispatchEvent(new Event('input', { bubbles: true }));
        });
    }

    function initShortfallPresets() {
        $$('.sf-preset-chip').forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = SHIFT_PRESETS[btn.dataset.preset];
                if (!preset) return;
                applyPreset('sf', preset);
                $$('.sf-preset-chip').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    function renderShortfallResults(data) {
        const area = $('#shortfallResults');
        if (!area) return;
        area.style.display = 'block';

        // Risk Badge
        const badge = $('#riskTierBadge');
        if (badge) {
            badge.className = `risk-pill-badge risk-${data.risk_tier.toLowerCase()}`;
            badge.textContent = `${data.risk_tier} OPERATIONAL RISK`;
        }

        // 4 KPI Metrics
        if ($('#resEfficiency')) $('#resEfficiency').textContent = `${(data.predicted_efficiency * 100).toFixed(1)}%`;
        if ($('#resPredictedProd')) $('#resPredictedProd').textContent = `${data.predicted_production.toFixed(1)} T`;
        if ($('#resTargetSub')) $('#resTargetSub').textContent = `Target: ${data.target_production.toLocaleString()} T`;
        if ($('#resShortfallTonnes')) $('#resShortfallTonnes').textContent = `${(data.target_production - data.predicted_production).toFixed(1)} T`;
        if ($('#resShortfallPct')) $('#resShortfallPct').textContent = `${data.shortfall_pct.toFixed(1)}% Shortfall`;
        if ($('#resRiskFlags')) $('#resRiskFlags').textContent = `${data.risk_flags} Detected`;

        renderRootCauseChart($('#rootCausesContainer'), data.root_causes);

        // Prescriptive Engineering Directives
        const recBox = $('#recommendationsContainer');
        if (recBox) {
            recBox.innerHTML = '';
            (data.recommendations || []).forEach(rec => {
                const card = document.createElement('div');
                const isCritical = rec.toLowerCase().includes('escalate') || rec.toLowerCase().includes('exceeds');
                const isWarning = rec.toLowerCase().includes('preventive') || rec.toLowerCase().includes('drainage');
                card.className = `directive-card ${isCritical ? 'directive-critical' : (isWarning ? 'directive-warning' : '')}`;
                card.innerHTML = `
                    <div class="directive-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            ${isCritical ? '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>' : '<polyline points="9 11 12 14 22 4"></polyline><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>'}
                        </svg>
                    </div>
                    <span>${rec}</span>
                `;
                recBox.appendChild(card);
            });
        }

        area.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Module 4: Simulator Ledger ───────────────────────────
    function initSimulatorEngine() {
        setupRangeSlider('sim-equipment_availability', 'sim-val-equip-avail', v => `${v}`);
        setupRangeSlider('sim-drilling_delay', 'sim-val-drill-delay', v => `${v} hrs`);
        setupRangeSlider('sim-rainfall', 'sim-val-rainfall', v => `${v} mm`);
        setupRangeSlider('sim-soil_moisture', 'sim-val-soil-moisture', v => `${v}`);

        $('#simulatorForm')?.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = $('#btnRunSimulation');
            setLoading(btn, true);

            // /simulate now compares TWO states: the shortfall form is the
            // baseline (current operating conditions), the simulator form is
            // the scenario. If the shortfall form isn't on the page, fall back
            // to comparing the scenario against itself so nothing crashes.
            const scenario = gatherFormData('sim');
            const baseline = gatherFormDataSafe('sf', scenario);
            const payload = {
                mode: $('#sim-mode') ? $('#sim-mode').value : 'what_if',
                baseline: baseline,
                scenario: scenario,
            };
            try {
                const data = await apiPost('/simulate', payload);
                renderSimulatorResults(data);
                addSimToHistory(data);
            } catch (err) {
                showToast('Simulation error: ' + err.message, 'error');
            } finally {
                setLoading(btn, false);
            }
        });
    }

    function renderSimulatorResults(data) {
        const area = $('#simResults');
        if (!area) return;
        area.style.display = 'block';

        const base = data.baseline;
        const scen = data.scenario;
        const d = data.delta;

        const badge = $('#simRiskBadge');
        if (badge) {
            badge.className = `risk-pill-badge risk-${scen.risk_tier.toLowerCase()}`;
            badge.textContent = `${scen.risk_tier} SCENARIO RISK`;
        }

        if ($('#simEfficiency')) $('#simEfficiency').textContent = `${(scen.predicted_efficiency * 100).toFixed(1)}%`;
        if ($('#simPredictedProd')) $('#simPredictedProd').textContent = `${scen.predicted_production.toFixed(1)} T`;
        if ($('#simShortfallTonnes')) $('#simShortfallTonnes').textContent = `${scen.shortfall_tonnes.toFixed(1)} T`;
        if ($('#simShortfallPct')) $('#simShortfallPct').textContent = `${scen.shortfall_pct.toFixed(1)}% Target Shortfall`;

        // Plain-English verdict from the backend
        const sumBox = $('#simSummary');
        if (sumBox) {
            const improving = d.production_change_tonnes > 0;
            sumBox.className = `sim-summary ${improving ? 'sim-summary-good' : (d.production_change_tonnes < 0 ? 'sim-summary-bad' : '')}`;
            sumBox.textContent = data.summary;
        }

        // Baseline vs scenario comparison table
        const cmp = $('#simComparison');
        if (cmp) {
            // Colour follows the SIGN, not the meaning: positive green,
            // negative red, zero neutral. Note this means a growing shortfall
            // (+ T) reads green even though it is a worse outcome — the third
            // argument is kept only so the call sites stay self-documenting.
            const signed = (v, unit, _higherIsBetter, decimals = 1) => {
                let cls = 'delta-flat';
                if (v > 0) cls = 'delta-good';
                else if (v < 0) cls = 'delta-bad';
                return `<span class="${cls}">${v > 0 ? '+' : ''}${v.toFixed(decimals)}${unit}</span>`;
            };
            cmp.innerHTML = `
                <table class="cmp-table">
                    <thead>
                        <tr><th>Metric</th><th>Baseline</th><th>Scenario</th><th>Change</th></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Efficiency</td>
                            <td class="mono-val">${(base.predicted_efficiency * 100).toFixed(1)}%</td>
                            <td class="mono-val">${(scen.predicted_efficiency * 100).toFixed(1)}%</td>
                            <td class="mono-val">${signed(d.efficiency_change_pct_points, ' pp', true)}</td>
                        </tr>
                        <tr>
                            <td>Production</td>
                            <td class="mono-val">${base.predicted_production.toFixed(1)} T</td>
                            <td class="mono-val">${scen.predicted_production.toFixed(1)} T</td>
                            <td class="mono-val">${signed(d.production_change_tonnes, ' T', true)}</td>
                        </tr>
                        <tr>
                            <td>Shortfall</td>
                            <td class="mono-val">${base.shortfall_tonnes.toFixed(1)} T</td>
                            <td class="mono-val">${scen.shortfall_tonnes.toFixed(1)} T</td>
                            <td class="mono-val">${signed(d.shortfall_change_tonnes, ' T', false)}</td>
                        </tr>
                        <tr>
                            <td>Risk tier</td>
                            <td>${base.risk_tier}</td>
                            <td>${scen.risk_tier}</td>
                            <td>${d.risk_tier_changed ? d.risk_tier_change : 'unchanged'}</td>
                        </tr>
                        <tr>
                            <td>Risk flags</td>
                            <td class="mono-val">${base.risk_flags}</td>
                            <td class="mono-val">${scen.risk_flags}</td>
                            <td class="mono-val">${signed(d.risk_flags_change, '', false, 0)}</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }

        renderRootCauseChart($('#simRootCauses'), scen.root_causes);
    }

    function addSimToHistory(data) {
        state.simHistory.unshift(data);
        if (state.simHistory.length > 5) state.simHistory.pop();

        const tbody = $('#historyBody');
        if (!tbody) return;
        tbody.innerHTML = '';

        state.simHistory.forEach((run, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${state.simHistory.length - idx}</td>
                <td>${run.scenario.target_production.toLocaleString()} T</td>
                <td>${run.scenario.predicted_production.toFixed(1)} T</td>
                <td style="color:${run.scenario.shortfall_pct > 15 ? 'var(--risk-high)' : 'var(--text-secondary)'}; font-weight:700;">
                    ${run.scenario.shortfall_pct.toFixed(1)}%
                </td>
                <td>${(run.scenario.predicted_efficiency * 100).toFixed(1)}%</td>
                <td><span class="stat-chip chip-${run.scenario.risk_tier === 'LOW' ? 'success' : (run.scenario.risk_tier === 'MEDIUM' ? 'cyan' : 'purple')}">${run.scenario.risk_tier}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // ── Global Bootstrapper ──────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        initNavigation();
        initDashboard();
        initReserveMap();
        initShortfallEngine();
        initShortfallPresets();
        initSimulatorEngine();

        // Restore the section named in the URL (#simulator, #reserve, ...).
        // Runs last so every module has initialised and the Leaflet map can
        // size itself correctly if we land straight on the prospecting view.
        const fromHash = window.location.hash.replace('#', '');
        showSection(VALID_SECTIONS.includes(fromHash) ? fromHash : 'dashboard');

        // Browser back/forward between sections.
        window.addEventListener('hashchange', () => {
            const sec = window.location.hash.replace('#', '');
            if (VALID_SECTIONS.includes(sec) && sec !== state.currentSection) {
                showSection(sec, false);
            }
        });
    });

})();
