

'use strict';


let state = {};   
let current = null; 
let scoringData = {};   
let scoringGroups = {};   
let addConceptTarget = null; 
let scoringDirty = false;
let previewWarningVisible = false;


function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

let _toastTimer;
function toast(msg, type = 'ok') {
    const el = document.getElementById('toast');
    const icons = { ok: '✓', err: '✕', info: 'ℹ' };
    el.innerHTML = `<span style="font-weight:700;font-size:14px;">${icons[type] || '✓'}</span> ${String(msg).replace(/</g, '&lt;')}`;
    el.className = `show ${type}`;
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { el.className = ''; }, 2800);
}

async function api(method, path, body) {
    const opts = { method };
    if (body !== undefined) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
    }

    let r;
    try {
        r = await fetch(path, opts);
    } catch (error) {
        const message = error instanceof Error ? error.message : 'Request failed.';
        toast(message, 'err');
        throw error;
    }

    const text = await r.text();
    if (!r.ok) {
        let message = text;
        try {
            const payload = JSON.parse(text);
            if (payload && typeof payload.error === 'string') {
                message = payload.error;
            }
        } catch (_) {
        }
        toast(message, 'err');
        throw new Error(message);
    }

    if (!text) return null;

    try {
        return JSON.parse(text);
    } catch (_) {
        const message = 'Invalid server response.';
        toast(message, 'err');
        throw new Error(message);
    }
}


function getKnownConceptGroupIds() {
    return new Set(Object.keys(state || {}));
}

function findUnknownConceptGroupIds(groups) {
    const knownGroupIds = getKnownConceptGroupIds();
    if (!knownGroupIds.size) return [];

    const invalid = [];
    for (const [dim, groupIds] of Object.entries(groups || {})) {
        for (const groupId of groupIds || []) {
            if (!knownGroupIds.has(groupId)) {
                invalid.push({ dim, groupId });
            }
        }
    }
    return invalid;
}

function formatUnknownConceptGroupIds(invalid) {
    const byDim = new Map();
    for (const { dim, groupId } of invalid) {
        if (!byDim.has(dim)) byDim.set(dim, []);
        byDim.get(dim).push(groupId);
    }

    return Array.from(byDim.entries())
        .map(([dim, groupIds]) => `${dim}: ${Array.from(new Set(groupIds)).sort().join(', ')}`)
        .join('; ');
}

function validateScoringGroupsOrToast(groups, actionLabel) {
    const invalid = findUnknownConceptGroupIds(groups);
    if (!invalid.length) return true;
    toast(`${actionLabel} blocked. Unknown concept group IDs for ${formatUnknownConceptGroupIds(invalid)}.`, 'err');
    return false;
}


function setPreviewUnsavedNotice(visible) {
    const el = document.getElementById('preview-unsaved-note');
    if (!el) return;
    el.textContent = visible
        ? 'Preview includes unsaved scoring changes. Save All to persist before closing.'
        : '';
    el.classList.toggle('show', visible);
}

function resetScoringDirtyState() {
    scoringDirty = false;
    previewWarningVisible = false;
    setPreviewUnsavedNotice(false);
}

function markScoringDirty() {
    scoringDirty = true;
    if (!previewWarningVisible) setPreviewUnsavedNotice(false);
}

function showPreviewUnsavedWarning() {
    previewWarningVisible = true;
    setPreviewUnsavedNotice(true);
    toast('Preview includes unsaved changes. Save All to persist.', 'info');
}


function switchTab(tab) {
    document.getElementById('tab-concepts').style.display = tab === 'concepts' ? 'flex' : 'none';
    document.getElementById('tab-scoring').style.display = tab === 'scoring' ? 'flex' : 'none';
    document.getElementById('btn-concepts').classList.toggle('active', tab === 'concepts');
    document.getElementById('btn-scoring').classList.toggle('active', tab === 'scoring');
    if (tab === 'scoring' && !Object.keys(scoringData).length) loadScoring();
}


async function load(keepCurrent) {
    let nextState;
    try {
        nextState = await api('GET', '/api/concepts');
    } catch (_) {
        return;
    }
    state = nextState;
    renderSidebar();
    if (keepCurrent && current && state[current]) {
        renderWords();
    } else {
        const first = Object.keys(state).sort()[0];
        if (first) selectGroup(first); else showEmptyPane();
    }
}

function renderSidebar() {
    const q = document.getElementById('search').value.toLowerCase();
    const list = document.getElementById('group-list');
    const names = Object.keys(state).sort().filter(n => !q || n.includes(q));
    list.innerHTML = names.map(n => `
    <div class="group-item${n === current ? ' active' : ''}" data-g="${esc(n)}" tabindex="0"
         title="${esc(n)} (${state[n].length} words)">
      <div class="group-item-main">
        <span class="group-name">${esc(n)}</span>
      </div>
      <div class="group-actions">
        <span class="badge">${state[n].length}</span>
        <button class="group-del-btn" data-g="${esc(n)}" title="Delete concept group">×</button>
      </div>
    </div>`).join('');
    list.querySelectorAll('.group-item').forEach(el => {
        el.addEventListener('click', () => selectGroup(el.dataset.g));
        el.addEventListener('keydown', e => { if (e.key === 'Enter') selectGroup(el.dataset.g); });
    });
    list.querySelectorAll('.group-del-btn').forEach(el => {
        el.addEventListener('click', e => {
            e.stopPropagation();
            deleteGroup(el.dataset.g);
        });
    });
}

function selectGroup(name) {
    current = name;
    renderSidebar();
    renderWords();
}

function showEmptyPane() {
    const main = document.getElementById('main');
    main.innerHTML = `
    <div id="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
      <p>Select a concept group from the sidebar, or create a new one to get started.</p>
    </div>`;
}

function renderWords() {
    const container = document.getElementById('main');
    if (!current) { showEmptyPane(); return; }
    const words = state[current] || [];

    container.innerHTML = `
    <div id="main-header">
      <div>
        <div id="group-title">${esc(current)}</div>
        <div id="group-subtitle">${words.length} word${words.length !== 1 ? 's' : ''}</div>
      </div>
      <button class="btn-danger" id="delete-group-btn" title="Delete this group">Delete group</button>
    </div>
    <div id="word-area">
      <div id="add-word-row">
        <input id="new-word" placeholder="Add words (comma-separated)…" autocomplete="off">
        <button class="btn-add" id="add-word-btn">Add</button>
      </div>
      <div id="tags-wrap">
        ${words.length
            ? words.map(w => `
              <div class="tag">
                <span>${esc(w)}</span>
                <button class="tag-del" data-w="${esc(w)}" title="Remove &quot;${esc(w)}&quot;">×</button>
              </div>`).join('')
            : '<span style="color:var(--muted);font-size:13px;">No words yet — add some above.</span>'
        }
      </div>
    </div>`;

    document.getElementById('new-word').addEventListener('keydown', e => {
        if (e.key === 'Enter') addWords();
    });
    document.getElementById('add-word-btn').addEventListener('click', addWords);
    document.getElementById('delete-group-btn').addEventListener('click', deleteGroup);
    document.querySelectorAll('.tag-del').forEach(btn =>
        btn.addEventListener('click', () => removeWord(btn.dataset.w)));
}

async function addWords() {
    const input = document.getElementById('new-word');
    const words = input.value.split(',').map(s => s.trim()).filter(Boolean);
    if (!words.length || !current) return;
    await api('POST', '/api/add_words', { group: current, words });
    input.value = '';
    await load(true);
    toast(`Added ${words.length} word${words.length !== 1 ? 's' : ''}`, 'ok');
}

async function removeWord(word) {
    await api('POST', '/api/remove_word', { group: current, word });
    await load(true);
    toast(`"${word}" removed`, 'ok');
}

async function deleteGroup(groupName = current) {
    if (!groupName) return;
    if (!confirm(`Delete group "${groupName}" and all its words?`)) return;
    const deletingCurrent = groupName === current;
    const result = await api('POST', '/api/delete_group', { group: groupName });
    if (deletingCurrent) current = null;
    await load(!deletingCurrent);
    syncScoringGroupsAfterConceptGroupDelete(groupName);
    if (result.removed_references) {
        toast(
            `Group deleted and removed from ${result.affected_dimensions} scoring dimension${result.affected_dimensions !== 1 ? 's' : ''}`,
            'ok',
        );
        return;
    }
    toast('Group deleted', 'ok');
}


function openNewGroupModal() {
    document.getElementById('modal-name').value = '';
    document.getElementById('modal-words').value = '';
    openModal('modal-bg');
    setTimeout(() => document.getElementById('modal-name').focus(), 50);
}

async function confirmNewGroup() {
    const name = document.getElementById('modal-name').value.trim().toLowerCase().replace(/\s+/g, '_');
    const words = document.getElementById('modal-words').value.split(',').map(s => s.trim()).filter(Boolean);
    if (!name) { toast('Group name required', 'err'); return; }
    if (state[name]) { toast('Group already exists', 'err'); return; }
    await api('POST', '/api/add_group', { group: name, words });
    closeModal('modal-bg');
    await load(false);
    selectGroup(name);
    toast(`Group "${name}" created`, 'ok');
}


function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }


async function loadScoring() {
    let nextScoringData;
    let nextScoringGroups;
    try {
        [nextScoringData, nextScoringGroups] = await Promise.all([
            api('GET', '/api/get_scoring'),
            api('GET', '/api/get_scoring_groups'),
        ]);
    } catch (_) {
        return;
    }
    scoringData = nextScoringData;
    scoringGroups = nextScoringGroups;
    resetScoringDirtyState();
    renderDimensions();
    renderBlendWeights();
    updateScoringBadge();
}

function updateScoringBadge() {
    const badge = document.getElementById('scoring-badge');
    if (badge) badge.textContent = Object.keys(scoringGroups).length;
}

function syncScoringGroupsAfterConceptGroupDelete(groupName) {
    if (!Object.keys(scoringData).length) return;

    for (const [dim, groupIds] of Object.entries(scoringGroups)) {
        scoringGroups[dim] = (groupIds || []).filter(groupId => groupId !== groupName);
    }

    renderDimensions();
    updateScoringBadge();
}

function renderDimensions() {
    const el = document.getElementById('dim-list');
    const dims = Object.keys(scoringGroups);
    const weights = scoringData._NARRATIVE_WEIGHTS || {};

    if (!dims.length) {
        el.innerHTML = '<p style="color:var(--muted);font-size:13px;">No scoring dimensions yet. Click ＋ to create one.</p>';
        return;
    }

    el.innerHTML = dims.map(dim => {
        const weight = weights[dim] !== undefined ? weights[dim] : 0.10;
        const conceptGroups = scoringGroups[dim] || [];
        const chips = conceptGroups.map(c => `
      <span class="ctag">${esc(c)}
        <button class="ctag-del" data-action="remove-dim-concept" data-dim="${esc(dim)}" data-concept="${esc(c)}" title="Remove">×</button>
      </span>`).join('');
        return `
    <div class="dim-card" data-dim="${esc(dim)}">
      <div class="dim-header">
        <span class="dim-name">${esc(dim)}</span>
        <div class="dim-slider-wrap">
          <input type="range" class="weight-slider" min="0" max="1" step="0.01"
                 value="${weight}" data-action="dim-slide" data-dim="${esc(dim)}"
                 title="Narrative weight for '${esc(dim)}'">
          <span class="weight-val" id="wv-${esc(dim)}">${weight.toFixed(2)}</span>
        </div>
        <button class="btn-del-dim" data-action="delete-dim" data-dim="${esc(dim)}" title="Delete dimension">×</button>
      </div>
      <div class="dim-concepts" id="dc-${esc(dim)}">${chips || '<span style="color:var(--muted);font-size:12px;">No concept groups yet</span>'}</div>
      <button class="btn-add-concept" style="margin-top:8px;" data-action="open-add-concept" data-dim="${esc(dim)}">＋ Add concept group</button>
    </div>`;
    }).join('');
}

function renderBlendWeights() {
    const el = document.getElementById('blend-weights');
    const data = scoringData._BLEND_WEIGHTS || {};
    el.innerHTML = Object.entries(data).map(([k, v]) => `
    <div class="weight-row">
      <span class="weight-label" title="${esc(k)}">${esc(k)}</span>
      <input type="range" class="weight-slider" min="0" max="1" step="0.01"
             value="${v}" oninput="onBlendSlide(this,'${esc(k)}')">
      <span class="weight-val" id="bwv-${esc(k)}">${v.toFixed(2)}</span>
    </div>`).join('');
}

function onDimSlide(slider, dim) {
    const val = parseFloat(slider.value);
    document.getElementById('wv-' + dim).textContent = val.toFixed(2);
    if (!scoringData._NARRATIVE_WEIGHTS) scoringData._NARRATIVE_WEIGHTS = {};
    scoringData._NARRATIVE_WEIGHTS[dim] = val;
    markScoringDirty();
}

function onBlendSlide(slider, key) {
    const val = parseFloat(slider.value);
    document.getElementById('bwv-' + key).textContent = val.toFixed(2);
    if (!scoringData._BLEND_WEIGHTS) scoringData._BLEND_WEIGHTS = {};
    scoringData._BLEND_WEIGHTS[key] = val;
    markScoringDirty();
}

function removeDimConcept(dim, concept) {
    scoringGroups[dim] = (scoringGroups[dim] || []).filter(c => c !== concept);
    markScoringDirty();
    renderDimensions();
}

function deleteDim(dim) {
    if (!confirm(`Delete scoring dimension "${dim}"?`)) return;
    delete scoringGroups[dim];
    if (scoringData._NARRATIVE_WEIGHTS) delete scoringData._NARRATIVE_WEIGHTS[dim];
    markScoringDirty();
    renderDimensions();
    updateScoringBadge();
}

async function saveScoring() {
    if (!validateScoringGroupsOrToast(scoringGroups, 'Save')) return;
    await api('POST', '/api/save_scoring_bundle', {
        scoring: scoringData,
        groups: scoringGroups,
    });
    resetScoringDirtyState();
    toast('Saved to scoring.py & narrative_elements.py', 'ok');
}

async function previewScore() {
    const query = document.getElementById('preview-query').value.trim();
    const candidate = document.getElementById('preview-candidate').value.trim();
    if (!query || !candidate) { toast('Enter both messages first', 'err'); return; }
    if (!validateScoringGroupsOrToast(scoringGroups, 'Preview')) return;
    const r = await api('POST', '/api/preview_score', {
        query,
        candidate,
        scoring: scoringData,
        groups: scoringGroups,
    });
    if (scoringDirty) showPreviewUnsavedWarning();
    document.getElementById('score-result').textContent = r.score.toFixed(4);
}


function openAddDim() {
    document.getElementById('sdim-name').value = '';
    document.getElementById('sdim-concepts').value = '';
    openModal('sdim-modal-bg');
    setTimeout(() => document.getElementById('sdim-name').focus(), 50);
}

function closeAddDim() { closeModal('sdim-modal-bg'); }

function confirmAddDim() {
    const name = document.getElementById('sdim-name').value.trim().toLowerCase().replace(/\s+/g, '_');
    if (!name) { toast('Dimension name required', 'err'); return; }
    if (scoringGroups[name] !== undefined) { toast('Dimension already exists', 'err'); return; }
    const conceptGroups = document.getElementById('sdim-concepts').value
        .split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    if (!validateScoringGroupsOrToast({ [name]: conceptGroups }, 'Create')) return;
    scoringGroups[name] = conceptGroups;
    if (!scoringData._NARRATIVE_WEIGHTS) scoringData._NARRATIVE_WEIGHTS = {};
    scoringData._NARRATIVE_WEIGHTS[name] = 0.10;
    markScoringDirty();
    closeAddDim();
    renderDimensions();
    updateScoringBadge();
}


function openAddConcept(dim) {
    addConceptTarget = dim;
    document.getElementById('sconcept-dim-label').textContent = dim;
    document.getElementById('sconcept-input').value = '';
    openModal('sconcept-modal-bg');
    setTimeout(() => document.getElementById('sconcept-input').focus(), 50);
}

function closeAddConcept() { closeModal('sconcept-modal-bg'); }

function confirmAddConcept() {
    const dim = addConceptTarget;
    const conceptGroups = document.getElementById('sconcept-input').value
        .split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    if (!conceptGroups.length) { toast('Enter at least one concept group ID', 'err'); return; }
    const updatedGroups = [...new Set([...(scoringGroups[dim] || []), ...conceptGroups])];
    if (!validateScoringGroupsOrToast({ [dim]: updatedGroups }, 'Add')) return;
    scoringGroups[dim] = updatedGroups;
    markScoringDirty();
    closeAddConcept();
    renderDimensions();
    toast(`Added ${conceptGroups.length} concept group${conceptGroups.length !== 1 ? 's' : ''} to "${dim}"`, 'ok');
}


document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        ['modal-bg', 'sdim-modal-bg', 'sconcept-modal-bg'].forEach(id => {
            const el = document.getElementById(id);
            if (el && el.classList.contains('open')) closeModal(id);
        });
    }
});


document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', e => {
        if (e.target === overlay) closeModal(overlay.id);
    });
});

const dimList = document.getElementById('dim-list');
dimList.addEventListener('click', e => {
    const actionEl = e.target.closest('[data-action]');
    if (!actionEl || !dimList.contains(actionEl)) return;

    const { action, dim, concept } = actionEl.dataset;
    if (action === 'remove-dim-concept') {
        removeDimConcept(dim, concept);
    } else if (action === 'delete-dim') {
        deleteDim(dim);
    } else if (action === 'open-add-concept') {
        openAddConcept(dim);
    }
});

dimList.addEventListener('input', e => {
    const slider = e.target.closest('[data-action="dim-slide"]');
    if (!slider || !dimList.contains(slider)) return;
    onDimSlide(slider, slider.dataset.dim);
});


document.getElementById('add-group-btn').addEventListener('click', openNewGroupModal);
document.getElementById('search').addEventListener('input', renderSidebar);


document.getElementById('modal-cancel').addEventListener('click', () => closeModal('modal-bg'));
document.getElementById('modal-ok').addEventListener('click', confirmNewGroup);
document.getElementById('modal-name').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('modal-words').focus(); });
document.getElementById('modal-words').addEventListener('keydown', e => { if (e.key === 'Enter') confirmNewGroup(); });

load(false);
