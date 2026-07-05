let freshCount = 0;
let rottenCount = 0;
let lastLabel = "Unknown";
let labelFrames = 0;

async function fetchTelemetry() {
    try {
        const response = await fetch('/api/telemetry');
        const data = await response.json();
        
        // Update Status
        const currentLabelEl = document.getElementById('current-label');
        const currentLabelDot = document.getElementById('current-label-dot');
        
        currentLabelEl.innerText = data.current_label.toUpperCase();
        
        if (data.current_label === "Fresh") {
            currentLabelDot.className = "glow-dot success active";
            currentLabelEl.style.color = "var(--success-emerald)";
        } else if (data.current_label === "Rotten") {
            currentLabelDot.className = "glow-dot error active";
            currentLabelEl.style.color = "var(--error-red)";
        } else {
            currentLabelDot.className = "glow-dot active";
            currentLabelEl.style.color = "var(--text-primary)";
        }
        
        // Update Confidence
        document.getElementById('current-confidence').innerText = data.current_confidence + "%";
        
        // Basic logic to increment counters if the object stays the same for 5 polls (~2.5s)
        // This is a naive client-side tracking for demonstration
        if (data.current_label === lastLabel && data.current_label !== "Unknown") {
            labelFrames++;
            if (labelFrames === 3) { // Trigger after a brief hold
                if (data.current_label === "Fresh") {
                    freshCount++;
                    document.getElementById('fresh-count').innerText = freshCount;
                } else if (data.current_label === "Rotten") {
                    rottenCount++;
                    document.getElementById('rotten-count').innerText = rottenCount;
                }
            }
        } else {
            labelFrames = 0;
            lastLabel = data.current_label;
        }
        
    } catch (err) {
        console.error("Error fetching telemetry:", err);
        document.getElementById('system-status-dot').className = "glow-dot error active";
        document.getElementById('system-status').innerText = "SYSTEM ERROR";
    }
}

function resetStats() {
    freshCount = 0;
    rottenCount = 0;
    document.getElementById('fresh-count').innerText = 0;
    document.getElementById('rotten-count').innerText = 0;
}

function exportData() {
    alert("Exporting log... (Simulation)");
}

// Poll every 500ms
setInterval(fetchTelemetry, 500);
