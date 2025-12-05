/* 
  EduGuard AI - Script Logic
  v4.0 - includes Robust PDF Fixes
*/

document.addEventListener('DOMContentLoaded', () => {
    const loading = document.getElementById('loading-state');
    const formContainer = document.getElementById('form-fields');
    const predictionForm = document.getElementById('prediction-form');

    // 1. Fetch Config
    fetch('/info')
        .then(res => res.json())
        .then(data => {
            if (!data.features) throw new Error("No features data");
            buildForm(data.features, formContainer);
            loading.style.display = 'none';
            predictionForm.style.display = 'block';

            // Animation
            const groups = document.querySelectorAll('.form-group');
            groups.forEach((g, idx) => {
                g.style.opacity = '0';
                g.style.transform = 'translateY(10px)';
                g.style.transition = `all 0.5s ease ${idx * 0.05}s`;
                setTimeout(() => { g.style.opacity = '1'; g.style.transform = 'translateY(0)'; }, 100);
            });
        })
        .catch(err => {
            console.error(err);
            loading.textContent = "Error loading system.";
        });

    // 2. Submit Handler
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.querySelector('button[type="submit"]');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;margin:0"></div> Analyzing...';
        btn.disabled = true;

        const formData = new FormData(predictionForm);
        constinputs = {};
        const inputs = {};
        formData.forEach((value, key) => inputs[key] = value);

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: inputs })
            });
            const result = await res.json();

            setTimeout(() => {
                showResult(result);
                btn.innerHTML = originalText;
                btn.disabled = false;
            }, 800);

        } catch (err) {
            alert(err);
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });
});

function buildForm(features, container) {
    features.forEach(feat => {
        const group = document.createElement('div');
        group.className = 'form-group';

        const label = document.createElement('label');
        label.htmlFor = feat.name;
        label.innerHTML = `${getIcon(feat.name)} ${feat.label || feat.name}`;

        let input;
        if (feat.type === 'categorical') {
            input = document.createElement('select');
            input.id = feat.name; input.name = feat.name; input.required = true;
            input.innerHTML = '<option value="" disabled selected>Select Option</option>';
            feat.options.forEach(opt => {
                const o = document.createElement('option');
                o.value = opt; o.text = opt;
                input.appendChild(o);
            });
        } else {
            input = document.createElement('input');
            input.type = 'number'; input.step = 'any'; input.id = feat.name; input.name = feat.name; input.required = true;
            input.placeholder = (feat.min !== null && feat.max !== null) ? `${feat.min} - ${feat.max}` : "Enter value";
            if (feat.min !== null) input.min = feat.min;
            if (feat.max !== null) input.max = feat.max;
        }

        group.appendChild(label);
        group.appendChild(input);
        if (feat.desc) {
            const hint = document.createElement('small');
            hint.className = 'input-hint';
            hint.innerHTML = `<i class="fas fa-info-circle"></i> ${feat.desc}`;
            group.appendChild(hint);
        }
        container.appendChild(group);
    });
}

function getIcon(name) {
    name = name.toLowerCase();
    if (name.includes('grade') || name.startsWith('g')) return '<i class="fas fa-chart-bar"></i>';
    if (name.includes('age')) return '<i class="fas fa-user-clock"></i>';
    return '<i class="fas fa-pen"></i>';
}

function showResult(data) {
    const modal = document.getElementById('result-modal');
    modal.classList.remove('hidden');

    const prob = Math.round(data.dropout_probability * 100);
    document.getElementById('prob-text').textContent = `${prob}%`;

    // Risk Badge
    const badge = document.getElementById('risk-badge');
    badge.textContent = `${data.risk_level} Risk Level`;

    let color = '#00D26A';
    if (data.risk_level === 'Medium') color = '#FFB020';
    if (data.risk_level === 'High') color = '#FF4F4F';

    badge.style.background = color;
    badge.style.color = data.risk_level === 'Medium' ? '#000' : '#fff';

    // Chart
    const circle = document.getElementById('risk-path');
    circle.style.stroke = color;
    circle.style.strokeDasharray = `${prob}, 100`;

    // Suggestions
    const list = document.getElementById('suggestions-list');
    list.innerHTML = '';
    data.suggestions.forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        list.appendChild(li);
    });

    // Factors
    const factors = document.getElementById('key-factors-list');
    factors.innerHTML = '';
    if (data.key_factors && data.key_factors.length) {
        data.key_factors.forEach(f => {
            const width = Math.min(Math.max(f.importance * 100 * 3, 10), 100);
            factors.innerHTML += `
                <div>
                    <div class="factor-header">
                        <span>${f.name}</span>
                        <span style="color:${color}">${Math.round(f.importance * 100)}% Impact</span>
                    </div>
                    <div class="factor-bar-bg"><div class="factor-bar" style="width:${width}%;background:${color}"></div></div>
                </div>
            `;
        });
    }

    // Date
    document.getElementById('report-date').textContent = new Date().toLocaleString();
}

function closeResult() {
    document.getElementById('result-modal').classList.add('hidden');
}

function resetForm() {
    document.getElementById('prediction-form').reset();
}

// CRITICAL: PDF FUNCTION
async function downloadReport() {
    const element = document.getElementById('analysisReport');
    const btn = document.getElementById('btn-download');

    // Feedback
    const oldHtml = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
    btn.disabled = true;

    // Ensure element is visible and scrolled into view
    // Note: We use the scroll-wrapper for this
    element.scrollIntoView({ behavior: 'auto', block: 'start' });

    // Force a small wait ensuring layout is stable
    await new Promise(r => setTimeout(r, 500));

    const opt = {
        margin: 0.5,
        filename: `EduGuard_Risk_Report_${new Date().getTime()}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: true, scrollY: 0 },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save()
        .then(() => {
            btn.innerHTML = oldHtml;
            btn.disabled = false;
        })
        .catch(err => {
            console.error("PDF Fail:", err);
            alert("Error generating PDF: " + err.message);
            btn.innerHTML = oldHtml;
            btn.disabled = false;
        });
}
