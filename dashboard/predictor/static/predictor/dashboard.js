// Dashboard JavaScript — Chart.js charts + AJAX polling + server-side preset execution

(function () {
    const SENSOR_KEYS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
    ];

    // Track alert escalation and first CRITICAL for overlay verdict
    let previousAlert = 'HEALTHY';
    let failureShown = false;  // so MACHINE FAILED overlay only fires once per run
    let isBlockingPoll = false; // blocks poll() while a new run initializes

    // ── Chart: Risk Timeline ────────────────────────────
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const riskChart = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Risk Probability',
                data: [],
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231,76,60,0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Risk Probability', color: '#8b949e' },
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(255,255,255,0.08)' }
                },
                x: {
                    title: { display: true, text: 'Step', color: '#8b949e' },
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(255,255,255,0.08)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8b949e' } }
            }
        }
    });

    function addThresholdLines(chart, steps) {
        if (chart.data.datasets.length === 1) {
            chart.data.datasets.push({
                label: 'Warning (0.459)',
                data: [],
                borderColor: '#e67e22',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false,
            });
            chart.data.datasets.push({
                label: 'Critical (0.765)',
                data: [],
                borderColor: '#e74c3c',
                borderDash: [10, 5],
                pointRadius: 0,
                fill: false,
            });
        }
        const warningThreshold = 0.765 * 0.6;
        chart.data.datasets[1].data = steps.map(() => warningThreshold);
        chart.data.datasets[2].data = steps.map(() => 0.765);
    }

    // ── Charts: Individual Sensor Readings ──────────────────────────
    const sensorColors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#34495e'];
    const sensorLabels = ['Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear'];
    const sensorCanvasIds = ['chart-air', 'chart-proc', 'chart-rpm', 'chart-torque', 'chart-wear'];
    const sensorCharts = [];

    const commonSensorOptions = {
        responsive: true,
        scales: {
            y: {
                title: { display: true, text: 'Value', color: '#8b949e' },
                ticks: { color: '#8b949e' },
                grid: { color: 'rgba(255,255,255,0.08)' }
            },
            x: {
                title: { display: true, text: 'Step', color: '#8b949e' },
                ticks: { color: '#8b949e' },
                grid: { color: 'rgba(255,255,255,0.08)' }
            },
        },
        plugins: {
            legend: { labels: { color: '#8b949e' } }
        }
    };

    for (let i = 0; i < 5; i++) {
        const ctx = document.getElementById(sensorCanvasIds[i]).getContext('2d');
        sensorCharts.push(new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: sensorLabels[i],
                    data: [],
                    borderColor: sensorColors[i],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                }]
            },
            options: commonSensorOptions
        }));
    }

    // ── Update the UI from step log ─────────────────────
    function updateDashboard(log) {
        if (!log || log.length === 0) return;

        const latest = log[log.length - 1];
        const steps = log.map(s => s.step);

        const riskDisplay = document.getElementById('risk-display');
        riskDisplay.textContent = latest.risk_prob.toFixed(4);
        riskDisplay.className = 'risk-big ' + alertClass(latest.alert_level);

        const badge = document.getElementById('alert-badge');
        badge.textContent = latest.alert_level;
        badge.className = 'alert-badge ' + alertClass(latest.alert_level);

        document.getElementById('stat-risk').textContent = latest.risk_prob.toFixed(4);
        document.getElementById('stat-streak').textContent = latest.streak;
        document.getElementById('stat-failure').textContent = latest.failure_type || '\u2014';

        if (latest.sensors) {
            document.getElementById('s-air').textContent = latest.sensors['Air temperature [K]'];
            document.getElementById('s-proc').textContent = latest.sensors['Process temperature [K]'];
            document.getElementById('s-rpm').textContent = latest.sensors['Rotational speed [rpm]'];
            document.getElementById('s-torque').textContent = latest.sensors['Torque [Nm]'];
            document.getElementById('s-wear').textContent = latest.sensors['Tool wear [min]'];
        }

        if (latest.details) {
            document.getElementById('p-twf').textContent = latest.details.TWF;
            document.getElementById('p-hdf').textContent = latest.details.HDF;
            document.getElementById('p-pwf').textContent = latest.details.PWF;
            document.getElementById('p-osf').textContent = latest.details.OSF;
        }

        riskChart.data.labels = steps;

        let riskColor = '#27ae60'; // healthy
        let riskBg = 'rgba(39,174,96,0.1)';
        if (latest.alert_level === 'CRITICAL') {
            riskColor = '#e74c3c';
            riskBg = 'rgba(231,76,60,0.1)';
        } else if (latest.alert_level === 'WARNING') {
            riskColor = '#e67e22';
            riskBg = 'rgba(230,126,34,0.1)';
        }

        riskChart.data.datasets[0].data = log.map(s => s.risk_prob);
        riskChart.data.datasets[0].borderColor = riskColor;
        riskChart.data.datasets[0].backgroundColor = riskBg;

        addThresholdLines(riskChart, steps);
        riskChart.update();

        sensorCharts.forEach((chart, i) => {
            chart.data.labels = steps;
            chart.data.datasets[0].data = log.map(s => s.sensors ? s.sensors[SENSOR_KEYS[i]] : 0);
            chart.update();
        });

        const tbody = document.getElementById('log-body');
        tbody.innerHTML = '';
        for (let i = log.length - 1; i >= 0; i--) {
            const s = log[i];
            const tr = document.createElement('tr');
            const cls = alertClass(s.alert_level);
            tr.innerHTML = `
                <td>${s.step}</td>
                <td>${s.risk_prob.toFixed(4)}</td>
                <td class="${cls}">${s.alert_level}</td>
                <td>${s.failure_type || '\u2014'}</td>
                <td>${s.sensors ? s.sensors['Air temperature [K]'] : '\u2014'}</td>
                <td>${s.sensors ? s.sensors['Process temperature [K]'] : '\u2014'}</td>
                <td>${s.sensors ? s.sensors['Rotational speed [rpm]'] : '\u2014'}</td>
                <td>${s.sensors ? s.sensors['Torque [Nm]'] : '\u2014'}</td>
                <td>${s.sensors ? s.sensors['Tool wear [min]'] : '\u2014'}</td>
            `;
            tbody.appendChild(tr);
        }

        updateMachineStatus(latest.alert_level);
    }

    function alertClass(level) {
        if (level === 'CRITICAL') return 'critical';
        if (level === 'WARNING') return 'warning';
        return 'healthy';
    }

    function updateMachineStatus(alertLevel) {
        const box = document.getElementById('machine-status-box');
        const icon = document.getElementById('machine-status-icon');
        const label = document.getElementById('machine-status-label');
        if (!box || !icon || !label) return;

        box.classList.remove('warning', 'critical', 'failed');

        if (alertLevel === 'CRITICAL') {
            box.classList.add('critical');
            icon.textContent = '🔴';
            label.textContent = 'AT RISK';
        } else if (alertLevel === 'WARNING') {
            box.classList.add('warning');
            icon.textContent = '🟡';
            label.textContent = 'DEGRADING';
        } else {
            icon.textContent = '🟢';
            label.textContent = 'OPERATIONAL';
        }
    }

    // ── Show overlays ───────────────────────────────────
    function showCriticalOverlay(riskProb, failureType, step) {
        const overlay = document.getElementById('critical-overlay');
        const msg = document.getElementById('critical-msg');
        if (overlay && msg) {
            msg.textContent = 'Risk: ' + riskProb + ' | Failure: ' + (failureType || 'Unknown') + ' | Step: ' + step;
            overlay.style.display = 'flex';
            sessionStorage.setItem('criticalShownForStep', step);
        }
    }

    function showMachineFailed(failureStep, failureType, firstCriticalStep) {
        const overlay = document.getElementById('failed-overlay');
        const msg = document.getElementById('failed-msg');
        const verdict = document.getElementById('failed-verdict');
        if (overlay && msg && verdict) {
            msg.textContent = 'The machine failed at step ' + failureStep + (failureType ? ' (' + failureType + ')' : '');
            if (firstCriticalStep !== null && firstCriticalStep < failureStep) {
                verdict.textContent = '\u2705 System predicted failure ' + (failureStep - firstCriticalStep) + ' steps before it happened (step ' + firstCriticalStep + ')';
            } else {
                verdict.textContent = '\u274C System did not predict failure in advance';
            }
            overlay.style.display = 'flex';
            sessionStorage.setItem('failedShownForStep', failureStep);
        }
        const box = document.getElementById('machine-status-box');
        const icon = document.getElementById('machine-status-icon');
        const label = document.getElementById('machine-status-label');
        if (box && icon && label) {
            box.classList.remove('warning', 'critical');
            box.classList.add('failed');
            icon.textContent = '💀';
            label.textContent = 'FAILED';
        }
    }

    // ── Dual-tone alternating alarm (2200Hz / 1400Hz) ──
    // Alternates between two pitches like a European emergency alert.
    // The pitch jump creates urgency that a single tone never achieves.
    function playAlarm() {
        try {
            const AC = window.AudioContext || window.webkitAudioContext;
            const ctx = new AC();
            const FREQS = [2200, 1400];  // high / low alternating
            const ON_MS = 100;           // each tone length
            const GAP_MS = 30;            // silence between tones
            const PULSES = 8;             // total bursts (= ~2 full cycles)
            const GAIN = 0.9;

            for (let i = 0; i < PULSES; i++) {
                const freq = FREQS[i % 2];
                const start = ctx.currentTime + i * (ON_MS + GAP_MS) / 1000;
                const stop = start + ON_MS / 1000;

                const osc = ctx.createOscillator();
                const vol = ctx.createGain();
                osc.connect(vol);
                vol.connect(ctx.destination);
                osc.type = 'square';
                osc.frequency.value = freq;
                vol.gain.setValueAtTime(GAIN, start);
                vol.gain.setValueAtTime(0.001, stop);
                osc.start(start);
                osc.stop(stop + 0.01);
            }
        } catch (e) { }
    }


    // ── MP3 alert for CRITICAL events ───────────────────
    function playAlertSound() {
        try {
            const audio = new Audio('/static/predictor/alert.mp3');
            audio.volume = 1.0;
            audio.play().catch(() => { });
        } catch (e) { }
    }

    // ── Send one manual prediction step ─────────────────
    async function sendPredict(sensorValues) {
        const body = {};
        SENSOR_KEYS.forEach((k, i) => body[k] = sensorValues[i]);

        const res = await fetch('/api/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return await res.json();
    }

    // ── Start server-side preset (fire-and-forget) ──────
    // The server runs all steps in a background thread.
    // This page can navigate anywhere — the preset keeps running.
    // The regular poll loop below picks up all updates.
    async function runPreset(presetName) {
        isBlockingPoll = true;
        // Reset overlay state for new run
        failureShown = false;
        previousAlert = 'HEALTHY';
        stopAlarmLoop();

        const criticalOverlay = document.getElementById('critical-overlay');
        const failedOverlay = document.getElementById('failed-overlay');
        if (criticalOverlay) criticalOverlay.style.display = 'none';
        if (failedOverlay) failedOverlay.style.display = 'none';
        sessionStorage.removeItem('criticalShownForStep');
        sessionStorage.removeItem('failedShownForStep');

        // Reset charts so old lines don't flash
        riskChart.data.labels = [];
        riskChart.data.datasets[0].data = [];
        riskChart.update();

        sensorCharts.forEach(chart => {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update();
        });

        const res = await fetch('/api/run_preset/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset: presetName }),
        });
        const data = await res.json();
        isBlockingPoll = false;
        if (data.error) {
            console.error('Preset error:', data.error);
        }
        // No loop here — the poll() below does everything from this point on
    }

    // ── Poll for updates ────────────────────────────────
    // Runs every second while a preset is active, every 2 s otherwise.
    // Handles alert escalation overlays and the MACHINE FAILED overlay.
    let pollInterval = null;
    let alarmLoopTimer = null;  // repeats warning beep while at WARNING level

    function startAlarmLoop() {
        if (alarmLoopTimer) return;  // already running
        playAlarm();
        alarmLoopTimer = setInterval(playAlarm, 1800);  // repeat every 1.8 s
    }

    function stopAlarmLoop() {
        if (alarmLoopTimer) {
            clearInterval(alarmLoopTimer);
            alarmLoopTimer = null;
        }
    }

    async function poll() {
        if (isBlockingPoll) return;
        try {
            const res = await fetch('/api/log/');
            const data = await res.json();
            const log = data.log;
            const presetState = data.preset_state || {};
            const firstCriticalStep = data.first_critical_step;

            updateDashboard(log);

            // Alert escalation overlays (based on latest log entry)
            const latest = log && log[log.length - 1];
            if (latest) {
                if (latest.alert_level === 'WARNING') {
                    if (previousAlert !== 'WARNING') startAlarmLoop();  // just entered WARNING
                } else {
                    stopAlarmLoop();  // left WARNING — stop immediately
                    if (latest.alert_level === 'CRITICAL' && previousAlert !== 'CRITICAL') {
                        // Only show if not already seen in this session for this specific step
                        if (sessionStorage.getItem('criticalShownForStep') !== String(latest.step)) {
                            playAlertSound();
                            showCriticalOverlay(latest.risk_prob, latest.failure_type, latest.step);
                        }
                    }
                }
                previousAlert = latest.alert_level;
            }

            // Show MACHINE FAILED overlay once when preset finishes with a failure step
            if (!failureShown && presetState.done && presetState.failure_step) {
                if (sessionStorage.getItem('failedShownForStep') !== String(presetState.failure_step)) {
                    failureShown = true;
                    const failureType = latest ? latest.failure_type : null;
                    showMachineFailed(presetState.failure_step, failureType, firstCriticalStep);
                } else {
                    failureShown = true; // Mark as shown locally so we don't keep checking storage
                }
            }

            // Speed up polling while preset is running
            const isRunning = presetState.running;
            if (isRunning && pollInterval !== 1000) {
                clearInterval(_pollTimer);
                _pollTimer = setInterval(poll, 1000);
                pollInterval = 1000;
            } else if (!isRunning && pollInterval !== 2000) {
                clearInterval(_pollTimer);
                _pollTimer = setInterval(poll, 2000);
                pollInterval = 2000;
            }
        } catch (e) { /* ignore transient errors */ }
    }

    // ── Check for ?preset= query param on page load ─────
    const params = new URLSearchParams(window.location.search);
    const presetParam = params.get('preset');
    if (presetParam) {
        window.history.replaceState({}, '', '/');
        runPreset(presetParam);
    }

    // Start polling
    poll();
    let _pollTimer = setInterval(poll, 2000);
    pollInterval = 2000;

})();
