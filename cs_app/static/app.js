'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const pages = ['dashboard', 'fixtures', 'history', 'mappings', 'jobs', 'settings', 'research'];
  const activeJobs = new Set(['queued', 'running']);
  const retryableJobs = new Set(['partial', 'failed', 'budget_paused', 'cancelled']);
  const collectionKinds = new Set(['backfill', 'upcoming', 'finished', 'results']);
  const state = { page: '', status: null, settings: null, jobs: [], poll: null, detailId: null,
    fixtures: { offset: 0, limit: 50, version: 0 }, history: { offset: 0, limit: 50, version: 0 },
    mappings: { offset: 0, limit: 50, version: 0 }, odds: { offset: 0, limit: 100, version: 0 } };
  let detailVersion = 0;
  let confirmationResolve = null;

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function append(parent, ...nodes) { nodes.forEach((node) => { if (node != null) parent.append(node); }); return parent; }
  function clear(node) { node.replaceChildren(); }
  function message(node, text, error = false) {
    node.textContent = text || ''; node.classList.toggle('error-text', error);
  }
  function globalError(error) { $('global-error').hidden = false; message($('global-error'), error.message || String(error)); }
  function button(text, action, className = 'secondary') {
    const node = el('button', text, className); node.type = 'button';
    node.addEventListener('click', async () => {
      if (node.disabled) return; node.disabled = true;
      try { await action(); } catch (error) { globalError(error); }
      finally { if (node.isConnected) node.disabled = false; }
    });
    return node;
  }
  function link(text, href, download = false) {
    const node = el('a', text); node.href = href; if (download) node.setAttribute('download', ''); return node;
  }
  function shown(value) { return value === null || value === undefined || value === '' ? '—' : String(value); }
  function number(value, digits = 3) { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'; }
  function timestamp(value) {
    if (!value) return 'Unknown';
    if (!/(Z|[+-]\d{2}:\d{2})$/i.test(String(value))) return `${value} · timezone uncertain`;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? `${value} · invalid clock` : date.toISOString().replace('T', ' ').replace('.000Z', ' UTC').replace('Z', ' UTC');
  }
  function score(item) { return item && item.score1 != null && item.score2 != null ? `${item.score1} : ${item.score2}` : 'No observed result'; }
  function badge(value) {
    const text = shown(value);
    const node = el('span', text.replaceAll('_', ' '), 'badge');
    if (['completed', 'manual', 'automatic', 'active', 'selected', 'admitted'].includes(text)) node.classList.add('positive');
    if (['failed', 'rejected', 'inactive', 'cancelled'].includes(text)) node.classList.add('negative');
    if (['running', 'queued', 'partial', 'budget_paused', 'ambiguous'].includes(text)) node.classList.add('caution');
    return node;
  }
  function empty(container, text) { clear(container); container.append(el('p', text, 'empty-state')); }
  function table(headers, rows) {
    const result = el('table'); const head = el('thead'); const row = el('tr');
    headers.forEach((header) => { const cell = el('th', header); cell.scope = 'col'; row.append(cell); }); head.append(row);
    const body = el('tbody');
    rows.forEach((cells) => { const tr = el('tr'); cells.forEach((content) => { const td = el('td'); td.append(content instanceof Node ? content : document.createTextNode(shown(content))); tr.append(td); }); body.append(tr); });
    return append(result, head, body);
  }
  function tableWrap(headers, rows) { return append(el('div', null, 'table-wrap'), table(headers, rows)); }
  function facts(data) {
    const list = el('dl', null, 'facts');
    Object.entries(data || {}).forEach(([key, value]) => append(list, el('dt', key.replaceAll('_', ' ')), el('dd', typeof value === 'object' && value !== null ? JSON.stringify(value) : shown(value))));
    return list;
  }
  function jsonDetails(title, value) {
    const details = el('details'); return append(details, el('summary', title), el('pre', JSON.stringify(value, null, 2), 'json-data'));
  }
  function section(title) { const node = el('article', null, 'card'); node.append(el('h3', title)); return node; }
  function matchName(item) { return `${shown(item.participant1_name || item.team1)} vs ${shown(item.participant2_name || item.team2)}`; }
  function labeledMatch(item) { return append(el('div'), el('strong', matchName(item)), el('small', shown(item.tournament), 'muted block')); }

  async function api(path, method = 'GET', data) {
    const options = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } };
    if (method !== 'GET') { options.headers['X-CS-App'] = '1'; options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(data || {}); }
    const response = await fetch(path, options);
    const raw = await response.text(); let body;
    try { body = raw ? JSON.parse(raw) : {}; } catch (_) { throw new Error(`The local API returned a non-JSON response (HTTP ${response.status}).`); }
    if (!response.ok) {
      const detail = body.detail || body.error || body.message;
      const explanation = Array.isArray(detail) ? detail.map((item) => `${(item.loc || []).join('.')}: ${item.msg || 'Invalid value'}`).join('; ') : typeof detail === 'object' ? JSON.stringify(detail) : detail;
      throw new Error(explanation || `Request failed (HTTP ${response.status}).`);
    }
    return body;
  }
  function query(values) { return new URLSearchParams(Object.entries(values).filter(([, value]) => value !== '' && value !== null && value !== undefined)).toString(); }
  async function pending(form, statusNode, work) {
    const controls = Array.from(form.querySelectorAll('input, select, button, textarea'));
    const prior = controls.map((node) => node.disabled); controls.forEach((node) => { node.disabled = true; }); form.setAttribute('aria-busy', 'true');
    message(statusNode, 'Working…');
    try { await work(); } catch (error) { message(statusNode, error.message, true); }
    finally { controls.forEach((node, i) => { node.disabled = prior[i]; }); form.removeAttribute('aria-busy'); }
  }
  function confirmCost(text) {
    if ($('confirmation-dialog').open) return Promise.resolve(false);
    $('confirmation-message').textContent = text;
    $('confirmation-dialog').returnValue = 'cancel'; $('confirmation-dialog').showModal(); $('confirm-action').focus();
    return new Promise((resolve) => { confirmationResolve = resolve; });
  }
  $('confirmation-dialog').addEventListener('close', () => {
    if (confirmationResolve) { confirmationResolve($('confirmation-dialog').returnValue === 'confirm'); confirmationResolve = null; }
  });

  function pager(container, data, listState, load) {
    clear(container);
    const total = Number(data.total) || 0;
    const previous = button('Previous', () => { listState.offset = Math.max(0, listState.offset - listState.limit); return load(); });
    const next = button('Next', () => { listState.offset += listState.limit; return load(); });
    previous.disabled = listState.offset === 0; next.disabled = listState.offset + listState.limit >= total;
    append(container, el('span', total ? `${listState.offset + 1}–${Math.min(listState.offset + (data.items || []).length, total)} of ${total.toLocaleString()}` : '0 records'), previous, next);
  }
  function fillBooks(books) {
    const names = [...new Set([...(books || []).map((book) => typeof book === 'string' ? book : book.slug || book.bookmaker || book.name).filter(Boolean), ...(state.settings?.bookmakers || [])])].sort();
    document.querySelectorAll('.book-select').forEach((select) => {
      const value = select.value; clear(select); const all = el('option', 'All bookmakers'); all.value = ''; select.append(all);
      if (value && !names.includes(value)) names.push(value);
      names.forEach((name) => { const option = el('option', name); option.value = name; select.append(option); }); select.value = value;
    });
  }
  function populateSettings(settings) {
    const form = $('settings-form');
    Object.entries(settings).forEach(([key, value]) => { const input = form.elements.namedItem(key); if (!input) return;
      if (input.type === 'checkbox') input.checked = value === true; else input.value = Array.isArray(value) ? value.join(',') : value;
    });
    state.settings = settings;
  }
  function renderStatus(data) {
    state.status = data; if (!state.settings) populateSettings(data.settings || {});
    fillBooks(data.books); clear($('dashboard-counts'));
    const labels = { fixtures: 'Provider fixtures', historical_matches: 'Historical matches', odds_states: 'Source odds states', quotes: 'Quote observations', mapped_fixtures: 'Mapped fixtures' };
    Object.entries(labels).forEach(([key, label]) => {
      const card = el('article', null, 'stat-card'); append(card, el('p', label), el('strong', typeof data.counts?.[key] === 'number' ? data.counts[key].toLocaleString() : '—')); $('dashboard-counts').append(card);
    });
    const warnings = $('dashboard-warnings'); clear(warnings);
    const settings = data.settings || {};
    warnings.append(el('div', settings.scheduler_enabled ? `Scheduled collection is enabled. Worker interval: ${shown(settings.scheduler_interval_minutes)} minutes; provider calls may occur without this page open.` : 'Scheduler is OFF. Opening the application does not start collection. Queue a job explicitly, or authorize scheduling in settings.', settings.scheduler_enabled ? 'notice warning' : 'notice'));
    if (!data.credential_configured) warnings.append(el('div', 'Provider credentials are not configured. Local imports, mapping review and exports remain available; authenticated collection cannot run.', 'notice warning'));
    clear($('dashboard-quota'));
    $('dashboard-quota').append(facts({ billable_attempts_today: data.quota?.billable_today, local_daily_cap: data.quota?.billable_per_day, local_per_run_cap: settings.billable_per_run, maximum_http_requests_per_run: settings.max_requests }));
    clear($('dashboard-account'));
    if (data.provider_account) {
      const account = data.provider_account;
      $('dashboard-account').append(facts({ active: account.active, request_count: account.request_count, request_limit: account.request_limit, entitled_bookmakers: account.bookmaker_slugs?.length, checked_at_UTC: timestamp(account.checked_at) }), jsonDetails('Full sanitized account snapshot', account));
    } else $('dashboard-account').append(el('p', 'No observed provider account snapshot yet.', 'muted'));
    renderJobs($('dashboard-jobs'), data.recent_jobs || [], false);
    $('jobs-warning').textContent = `${data.credential_configured ? 'Provider credentials are configured.' : 'Provider credentials are missing; collection will fail until configured.'} Scheduler ${settings.scheduler_enabled ? 'ON' : 'OFF'}. Saved local caps: ${shown(settings.billable_per_run)} billable attempts/run and ${shown(settings.billable_per_day)}/UTC day. All provider actions require cost authorization.`;
  }
  async function loadStatus() {
    message($('dashboard-state'), 'Loading workspace status…');
    try { renderStatus(await api('/api/status')); message($('dashboard-state'), `Status checked ${timestamp(new Date().toISOString())}`); }
    catch (error) { message($('dashboard-state'), error.message, true); }
  }
  function quoteCell(quotes) {
    const node = el('div', null, 'quote-stack');
    if (!quotes?.length) return el('span', 'No observations', 'muted');
    quotes.forEach((quote) => { const row = el('div');
      append(row, el('strong', `${shown(quote.bookmaker)} · ${shown(quote.kind)}`), el('span', ` ${number(quote.price1)} / ${number(quote.price2)}`), badge(quote.status), el('small', `${quote.reason || 'No reason reported'} · receipt ${timestamp(quote.observed_at)}`, 'muted block'));
      node.append(row);
    }); return node;
  }
  async function loadFixtures(mode = 'fixtures') {
    const listState = state[mode]; const version = ++listState.version;
    const form = $(mode === 'fixtures' ? 'fixture-filters' : 'mapping-filters');
    const values = Object.fromEntries(new FormData(form));
    const statusNode = $(`${mode}-state`); const container = $(`${mode}-table`);
    form.querySelector('button[type="submit"]').disabled = true; clear($(`${mode}-pager`));
    message(statusNode, 'Loading provider fixtures…'); container.setAttribute('aria-busy', 'true');
    try {
      const data = await api(`/api/fixtures?${query({ status: 'all', ...values, offset: listState.offset, limit: listState.limit })}`);
      if (version !== listState.version) return; clear(container);
      if (!data.items?.length) empty(container, 'No fixtures match these filters. Import existing archives or explicitly queue collection to add evidence.');
      else container.append(table(['Fixture / tournament', 'Scheduled start · UTC', 'Status / result', 'Mapping', mode === 'fixtures' ? 'Latest observed quotes' : 'Decision / reason', 'Review'], data.items.map((item) => [
        append(el('div'), labeledMatch(item), el('small', item.fixture_id, 'mono muted block')), timestamp(item.start_time), append(el('div'), badge(({ 0: 'upcoming', 1: 'live', 2: 'finished', 3: 'cancelled' })[item.status_id] || `status ${shown(item.status_id)}`), el('small', score(item.native_result), 'block muted')),
        append(el('div'), badge(item.mapping?.status || 'unmatched'), el('small', item.mapping?.history_key || 'No accepted history identity', 'mono muted block')),
        mode === 'fixtures' ? quoteCell(item.quotes) : append(el('div'), el('span', item.mapping?.reason || 'No review reason'), el('small', item.mapping?.orientation === 1 ? 'Same team order' : item.mapping?.orientation === 0 ? 'Reversed team order' : 'Orientation not established', 'muted block')),
        button(mode === 'fixtures' ? 'Open evidence' : 'Review mapping', () => openFixture(item.fixture_id, mode === 'mappings'))
      ])));
      pager($(`${mode}-pager`), data, listState, () => loadFixtures(mode)); message(statusNode, `${data.total} matching fixtures.`);
    } catch (error) { if (version === listState.version) { empty(container, 'Fixture data could not be loaded. Apply filters again to retry.'); clear($(`${mode}-pager`)); message(statusNode, error.message, true); } }
    finally { if (version === listState.version) { container.removeAttribute('aria-busy'); form.querySelector('button[type="submit"]').disabled = false; } }
  }
  function updateImportHint() {
    const hints = {
      opencs2: 'OpenCS2 maps.csv headers: match_id, match_url, match_date (timezone-aware), team1, team2, event, format, score1, score2, winner. Preserve all original map columns. Series labels require consistent explicit BO3 result metadata, not guessed map totals.',
      kaggle: 'Kaggle results_only.csv headers: hltv_match_id (native integer ID), date (timezone-aware), team1_name, team2_name, tournament, match_type, score_team1, score_team2, winner. Valid series labels require an explicit BO3 result.',
      strandedpond: 'StrandedPond matches.csv headers: match_url (native HLTV URL), date, team_left, team_right, event. Preserve all original map columns. Source dates remain timezone-uncertain and map-only records do not produce guessed series labels.',
      ektarr: 'Kaggle (ektarr) matches.csv headers: match_id, datetime (UTC), team1, team2, tournament, score1, score2, winner, format, games_played, bestOf. Explicit series results across all competitive tiers.'
    };
    $('history-import-hint').textContent = hints[$('history-import-source').value];
  }
  $('history-import-source').addEventListener('change', updateImportHint);
  $('history-import-form').addEventListener('submit', (event) => {
    event.preventDefault(); const form = event.currentTarget; const file = form.elements.namedItem('file').files[0];
    if (!file || !file.name.toLowerCase().endsWith('.csv')) { message($('history-import-state'), 'Choose a .csv file.', true); return; }
    if (file.size > 16 * 1024 * 1024) { message($('history-import-state'), 'The CSV exceeds the 16 MiB upload limit.', true); return; }
    if (!file.size) { message($('history-import-state'), 'The CSV file is empty.', true); return; }
    const source = form.elements.namedItem('source').value;
    pending(form, $('history-import-state'), async () => {
      const result = await api('/api/history/import', 'POST', { source, csv: await file.text() });
      state.importJob = result.id; state.importJobStatus = 'queued';
      message($('history-import-state'), `Import queued as job ${result.id}. Review collection jobs for progress, validation errors, or immutable-ID conflicts.`);
      form.elements.namedItem('file').value = ''; await loadJobs();
    });
  });
  updateImportHint();
  async function loadHistory() {
    const version = ++state.history.version; message($('history-state'), 'Loading historical matches…'); $('history-table').setAttribute('aria-busy', 'true');
    $('history-filters').querySelector('button[type="submit"]').disabled = true; clear($('history-pager'));
    try {
      const data = await api(`/api/history?${query({ ...Object.fromEntries(new FormData($('history-filters'))), offset: state.history.offset, limit: state.history.limit })}`);
      if (version !== state.history.version) return; clear($('history-table'));
      if (!data.items?.length) empty($('history-table'), 'No historical matches match these filters. The local import action adds the supported existing datasets.');
      else $('history-table').append(table(['Match / tournament', 'Source identity', 'Date evidence', 'Series result', 'Format / label'], data.items.map((item) => [labeledMatch(item), append(el('div'), el('strong', item.source), el('small', item.history_key, 'mono block')), append(el('div'), el('span', item.start_time ? timestamp(item.start_time) : shown(item.source_date)), badge(item.timezone_status || 'unknown')), score(item), append(el('div'), el('span', shown(item.format)), badge(item.label_status || 'unknown'))])));
      const sources = [...new Set((data.items || []).map((item) => item.source).filter(Boolean))];
      sources.forEach((source) => { if (![...$('history-sources').children].some((option) => option.value === source)) { const option = el('option'); option.value = source; $('history-sources').append(option); } });
      pager($('history-pager'), data, state.history, loadHistory); message($('history-state'), `${data.total} matching historical records.`);
    } catch (error) { if (version === state.history.version) { empty($('history-table'), 'Historical records could not be loaded. Apply filters again to retry.'); clear($('history-pager')); message($('history-state'), error.message, true); } }
    finally { if (version === state.history.version) { $('history-table').removeAttribute('aria-busy'); $('history-filters').querySelector('button[type="submit"]').disabled = false; } }
  }

  async function openFixture(id, mappingFocus = false) {
    state.detailId = String(id); state.odds.offset = 0; state.odds.version++;
    const version = ++detailVersion; clear($('detail-content')); $('detail-title').textContent = `Fixture ${id}`;
    message($('detail-state'), 'Loading fixture and evidence…'); if (!$('fixture-dialog').open) $('fixture-dialog').showModal();
    try {
      const item = await api(`/api/fixtures/${encodeURIComponent(id)}`);
      if (version !== detailVersion || !$('fixture-dialog').open) return;
      $('detail-title').textContent = matchName(item); message($('detail-state'), ''); renderDetail(item);
      if (mappingFocus) $('mapping-review').scrollIntoView({ block: 'start' });
    } catch (error) { if (version === detailVersion) message($('detail-state'), error.message, true); }
  }
  function renderDetail(item) {
    const target = $('detail-content'); clear(target);
    const overview = section('Provider-native fixture'); overview.append(facts({ fixture_id: item.fixture_id, tournament: item.tournament, scheduled_start_UTC: timestamp(item.start_time), actual_start_UTC: timestamp(item.true_start_time), status_id: item.status_id, participant_1: `${shown(item.participant1_name)} · ${shown(item.participant1_id)}`, participant_2: `${shown(item.participant2_name)} · ${shown(item.participant2_id)}`, native_series_result: score(item.native_result) }));
    overview.append(jsonDetails('Original fixture metadata', item.metadata || item.metadata_json || {})); target.append(overview);
    const quotes = section('Latest quote observations');
    quotes.append(el('p', 'The latest received state wins even if unusable. Source outcome order follows the provider catalog only; a quote is not certified executable.', 'muted'));
    if (!item.quotes?.length) quotes.append(el('p', 'No quote observations for this fixture.', 'empty-state'));
    else quotes.append(tableWrap(['Book / kind', 'Decimal 1 / 2', 'No-vig P(1)', 'Availability', 'Local receipt · UTC', 'Source clocks · UTC', 'Cutoff · UTC'], item.quotes.map((quote) => [ `${quote.bookmaker} · ${quote.kind}`, `${number(quote.price1)} / ${number(quote.price2)}`, typeof quote.probability1 === 'number' ? `${number(quote.probability1 * 100, 2)}%` : '—', append(el('div'), badge(quote.status), el('small', quote.reason || 'No reason reported', 'block muted')), timestamp(quote.observed_at), `${timestamp(quote.source_time_1)} / ${timestamp(quote.source_time_2)}`, timestamp(quote.cutoff_at) ])));
    target.append(quotes, mappingReview(item));
    const timeline = section('Normalized full Winner timeline'); timeline.id = 'odds-timeline';
    timeline.append(el('p', 'All normalized series Winner / player-0 states, including suspensions and malformed clocks. Historical source clock: createdAt; current source clock: changedAt. Receipt is the local observation time. Other markets remain in original raw evidence.', 'muted'));
    const form = el('form', null, 'filter-bar'); const label = el('label', 'Bookmaker'); const select = el('select'); select.name = 'bookmaker'; select.className = 'book-select'; label.append(select);
    const apply = el('button', 'Filter timeline'); apply.type = 'submit'; append(form, label, apply);
    const status = el('div', null, 'load-state'); status.id = 'odds-state'; status.setAttribute('role', 'status');
    const rows = el('div', null, 'table-wrap'); rows.id = 'odds-table'; const pagination = el('div', null, 'pager'); pagination.id = 'odds-pager';
    append(timeline, form, status, rows, pagination); target.append(timeline); fillBooks(state.status?.books);
    form.addEventListener('submit', (event) => { event.preventDefault(); state.odds.offset = 0; loadOdds(item.fixture_id, select.value); });
    loadOdds(item.fixture_id, '');
    const evidence = section('Raw evidence downloads'); evidence.append(el('p', 'Download the immutable gzip response. The decoded SHA-256 identifies the original provider bytes; download and verify independently.', 'muted'));
    if (!item.raw_evidence?.length) evidence.append(el('p', 'No saved raw responses linked to this fixture.', 'empty-state'));
    else evidence.append(tableWrap(['Endpoint / HTTP', 'Local receipt · UTC', 'Decoded SHA-256', 'Evidence'], item.raw_evidence.map((raw) => [ `${shown(raw.endpoint)} · ${shown(raw.http_status)}`, timestamp(raw.received_at), el('span', shown(raw.decoded_sha256), 'mono digest'), raw.decoded_sha256 ? link(`Download #${raw.id}`, `/api/evidence/${encodeURIComponent(raw.id)}`, true) : 'No raw body saved' ])));
    target.append(evidence);
  }
  async function loadOdds(id, bookmaker) {
    const version = ++state.odds.version; const status = $('odds-state'); const rows = $('odds-table'); const pagination = $('odds-pager'); if (!status) return;
    const apply = $('odds-timeline').querySelector('button[type="submit"]'); apply.disabled = true; clear(pagination);
    message(status, 'Loading source states…');
    try {
      const data = await api(`/api/fixtures/${encodeURIComponent(id)}/odds?${query({ bookmaker, offset: state.odds.offset, limit: state.odds.limit })}`);
      if (version !== state.odds.version || String(id) !== state.detailId || !rows.isConnected) return; clear(rows);
      if (!data.items?.length) empty(rows, 'No normalized Winner states in this scope. Inspect raw evidence for other markets or missingness.');
      else rows.append(table(['Bookmaker', 'Market / outcome', 'Source update · UTC', 'Local receipt · UTC', 'Active', 'Decimal price', 'Raw / state'], data.items.map((entry) => {
        const raw = entry.state || entry.state_json || {};
        const active = entry.active === true || entry.active === 1 ? 'active' : entry.active === false || entry.active === 0 ? 'inactive' : 'unknown';
        const details = jsonDetails('State data', raw);
        if (entry.raw_id != null) details.append(link(`Raw #${entry.raw_id}`, `/api/evidence/${encodeURIComponent(entry.raw_id)}`, true));
        return [entry.bookmaker, `${shown(entry.market_id)} / ${shown(entry.outcome_id)}`, timestamp(entry.source_time), timestamp(entry.observed_at || entry.received_at), badge(active), number(entry.price), details];
      })));
      pager(pagination, data, state.odds, () => loadOdds(id, bookmaker)); message(status, `${data.total} source states. Both clocks are shown independently.`);
    } catch (error) { if (version === state.odds.version && rows.isConnected) { empty(rows, 'Timeline could not be loaded. Filter again to retry.'); message(status, error.message, true); clear(pagination); } }
    finally { if (version === state.odds.version && apply.isConnected) apply.disabled = false; }
  }
  function mappingReview(item) {
    const card = section('Mapping review & decision history'); card.id = 'mapping-review';
    card.append(facts({ status: item.mapping?.status || 'unmatched', history_key: item.mapping?.history_key, orientation: item.mapping?.orientation === 1 ? 'Same order (provider 1 → historical 1)' : item.mapping?.orientation === 0 ? 'Reversed (provider 1 → historical 2)' : 'Unestablished', method: item.mapping?.method, reason: item.mapping?.reason }));
    const form = el('form', null, 'form-grid');
    const keyLabel = el('label', 'Historical identity (source:id)', 'full'); const key = el('input'); key.name = 'history_key'; key.value = item.mapping?.history_key || ''; key.placeholder = 'Choose a candidate or find a historical record'; keyLabel.append(key);
    const orientationLabel = el('label', 'Explicit team orientation'); const orientation = el('select'); orientation.name = 'orientation';
    [['', 'Select orientation'], ['1', 'Same: provider 1 = historical 1'], ['0', 'Reversed: provider 1 = historical 2']].forEach(([value, text]) => { const option = el('option', text); option.value = value; orientation.append(option); });
    orientation.value = item.mapping?.orientation == null ? '' : String(item.mapping.orientation); orientationLabel.append(orientation);
    const reasonLabel = el('label', 'Decision reason (required)', 'full'); const reason = el('textarea'); reason.name = 'reason'; reason.required = true; reason.rows = 3; reason.placeholder = 'Evidence for identity, source date, tournament, team order, or rejection.'; reasonLabel.append(reason);
    const decisionLabel = el('label', 'Decision'); const decision = el('select'); decision.name = 'decision';
    [['accept', 'Accept explicit link'], ['reject', 'Reject / retain exclusion'], ['clear', 'Clear override & conservatively remap']].forEach(([value, text]) => { const option = el('option', text); option.value = value; decision.append(option); }); decisionLabel.append(decision);
    const controls = el('div', null, 'form-actions full'); const submit = el('button', 'Save mapping decision'); submit.type = 'submit'; const status = el('span'); status.setAttribute('role', 'status'); append(controls, submit, status);
    append(form, keyLabel, orientationLabel, decisionLabel, reasonLabel, controls);
    decision.addEventListener('change', () => { key.required = decision.value === 'accept'; orientation.required = decision.value === 'accept'; }); key.required = true; orientation.required = true;
    const choose = (record) => { key.value = record.history_key; decision.value = 'accept'; key.required = true; orientation.required = true; orientation.value = ''; key.focus(); message(status, `Selected ${record.history_key}. Verify orientation and supply a reason before accepting.`); };
    card.append(el('h4', 'Candidates'));
    if (!item.candidates?.length) card.append(el('p', 'No conservative candidates. Search the historical corpus below to choose an evidenced identity manually.', 'empty-state'));
    else card.append(tableWrap(['Historical identity / match', 'Source date / result', 'Candidate evidence', 'Select'], item.candidates.map((candidate) => {
      const history = candidate.history || candidate.historical || candidate;
      return [append(el('div'), el('strong', candidate.history_key), labeledMatch(history)), append(el('div'), el('span', history.start_time ? timestamp(history.start_time) : shown(history.source_date)), el('small', `${score(history)} · ${shown(history.label_status)}`, 'block muted')), append(el('div'), badge(candidate.status), el('p', candidate.reason || ''), jsonDetails('Candidate details', candidate.details || candidate.details_json || {})), button('Use identity', () => choose(candidate))];
    })));
    const search = el('form', null, 'filter-bar'); const searchLabel = el('label', 'Find historical identity', 'grow'); const input = el('input'); input.type = 'search'; input.required = true; input.placeholder = 'Search teams, tournament or source identity'; searchLabel.append(input); const searchButton = el('button', 'Search history'); searchButton.type = 'submit'; append(search, searchLabel, searchButton);
    const searchStatus = el('div', null, 'load-state'); searchStatus.setAttribute('role', 'status'); const searchResults = el('div');
    search.addEventListener('submit', (event) => { event.preventDefault(); const text = input.value.trim(); if (!text) return;
      pending(search, searchStatus, async () => { clear(searchResults); const data = await api(`/api/history?${query({ q: text, limit: 20, offset: 0 })}`);
        if (!data.items?.length) empty(searchResults, 'No historical identity found.');
        else searchResults.append(tableWrap(['Match / identity', 'Date / result', 'Select'], data.items.map((record) => [append(el('div'), labeledMatch(record), el('small', record.history_key, 'mono block')), append(el('div'), el('span', record.start_time ? timestamp(record.start_time) : shown(record.source_date)), el('small', `${score(record)} · ${shown(record.timezone_status)}`, 'block muted')), button('Use identity', () => choose(record))])));
        message(searchStatus, `Showing ${data.items?.length || 0} of ${data.total} results. Refine the query if needed.`);
      });
    });
    append(card, search, searchStatus, searchResults, form);
    form.addEventListener('submit', (event) => { event.preventDefault(); const reasonText = reason.value.trim(); if (!reasonText) { message(status, 'A nonblank decision reason is required.', true); reason.focus(); return; }
      const payload = { decision: decision.value, reason: reasonText };
      if (key.value.trim()) payload.history_key = key.value.trim();
      if (decision.value === 'accept') { if (!payload.history_key || orientation.value === '') { message(status, 'Choose a historical identity and explicit team orientation.', true); return; } payload.orientation = Number(orientation.value); }
      pending(form, status, async () => { await api(`/api/mappings/${encodeURIComponent(item.fixture_id)}`, 'POST', payload); message(status, 'Decision saved.'); await openFixture(item.fixture_id, true); if (state.page === 'fixtures' || state.page === 'mappings') loadFixtures(state.page); loadStatus(); });
    });
    if (item.mapping_decisions?.length) card.append(jsonDetails('Audited previous mapping decisions', item.mapping_decisions));
    else card.append(el('p', 'No manual decision history.', 'muted'));
    return card;
  }

  function renderJobs(container, jobs, actions = true) {
    clear(container);
    if (!jobs.length) { empty(container, 'No jobs yet. Explicitly queue a collection, local import, or mapping rebuild.'); return; }
    jobs.forEach((job) => {
      const card = el('article', null, 'job-card'); const heading = el('div', null, 'section-heading'); append(heading, el('h3', job.kind?.replaceAll('_', ' ') || 'Job'), badge(job.status)); card.append(heading);
      card.append(el('small', job.id, 'mono muted block'));
      card.append(el('p', `Created ${timestamp(job.created_at)}${job.finished_at ? ` · finished ${timestamp(job.finished_at)}` : ''}`, 'muted'));
      if (job.cancel_requested) card.append(el('p', 'Cancellation requested; the worker will stop at the next safe boundary.', 'notice warning'));
      if (job.progress && Object.keys(job.progress).length) {
        const visibleProgress = Object.fromEntries(Object.entries(job.progress).filter(([, value]) => value == null || ['string', 'number', 'boolean'].includes(typeof value)));
        if (Object.keys(visibleProgress).length) card.append(facts(visibleProgress));
        card.append(jsonDetails('Full progress & request counters', job.progress));
      }
      if (job.result && Object.keys(job.result).length) card.append(jsonDetails('Observed result & coverage', job.result));
      if (job.payload && Object.keys(job.payload).length) card.append(jsonDetails('Frozen job scope', job.payload));
      if (job.error) card.append(el('p', job.error, 'error-text'));
      if (actions) {
        const controls = el('div', null, 'form-actions'); const status = el('span'); status.setAttribute('role', 'status');
        if (activeJobs.has(job.status)) {
          const cancel = button('Cancel job', async () => { await pending(controls, status, async () => { const result = await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, 'POST'); message(status, result.ok ? 'Cancellation requested.' : 'Job could not be cancelled in its current state.'); await loadJobs(); }); }); cancel.disabled = Boolean(job.cancel_requested); controls.append(cancel);
        }
        if (retryableJobs.has(job.status)) controls.append(button('Retry / resume job', async () => {
          if (collectionKinds.has(job.kind) && !state.settings) state.settings = await api('/api/settings');
          if (collectionKinds.has(job.kind) && !await confirmCost(`Retry ${job.kind} job ${job.id}? This resumes incomplete work and may make new billable provider requests within current saved caps (${shown(state.settings?.billable_per_run)} per run). Previously saved terminal history evidence may be reused.`)) return;
          await pending(controls, status, async () => { const result = await api(`/api/jobs/${encodeURIComponent(job.id)}/retry`, 'POST'); message(status, result.ok ? 'Job queued for retry.' : 'Job cannot be retried in its current state.'); await loadJobs(); });
        }));
        append(controls, button('Refresh detail', () => pending(controls, status, async () => {
          const updated = await api(`/api/jobs/${encodeURIComponent(job.id)}`); renderJobs(card.parentNode ? container : $('jobs-list'), state.jobs.map((current) => current.id === updated.id ? updated : current));
          state.jobs = state.jobs.map((current) => current.id === updated.id ? updated : current); schedulePoll();
        })), status); card.append(controls);
      }
      container.append(card);
    });
  }
  function schedulePoll() {
    clearTimeout(state.poll); state.poll = null;
    if (document.hidden || !state.jobs.some((job) => activeJobs.has(job.status))) return;
    state.poll = setTimeout(async () => { state.poll = null; const ok = await loadJobs(); if (ok) await loadStatus(); }, 4000);
  }
  function updateImportStatus() {
    if (!state.importJob) return;
    const job = state.jobs.find((item) => item.id === state.importJob); if (!job) return;
    const text = `Import ${job.id}: ${job.status.replaceAll('_', ' ')}${job.error ? ` · ${job.error}` : ''}`;
    message($('history-import-state'), text, ['failed', 'partial'].includes(job.status));
    if (job.status === 'completed' && state.importJobStatus !== 'completed' && state.page === 'history') loadHistory();
    state.importJobStatus = job.status;
  }
  async function loadJobs() {
    message($('jobs-state'), 'Loading jobs…');
    try { const data = await api('/api/jobs'); state.jobs = data.items || []; renderJobs($('jobs-list'), state.jobs); updateImportStatus(); message($('jobs-state'), state.jobs.some((job) => activeJobs.has(job.status)) ? 'Active jobs: status polling every 4 seconds while this tab is visible.' : 'No active jobs. Automatic polling is stopped.'); schedulePoll(); return true; }
    catch (error) { message($('jobs-state'), error.message, true); clearTimeout(state.poll); state.poll = null; return false; }
  }
  function updateJobForm() {
    const kind = $('job-kind').value; const collection = collectionKinds.has(kind); const backfill = kind === 'backfill';
    $('backfill-fields').hidden = !backfill; $('collection-fields').hidden = !collection;
    ['from', 'to'].forEach((name) => { const input = $('job-form').elements.namedItem(name); input.required = backfill; input.disabled = !backfill; });
    $('results-fields').hidden = kind !== 'results';
    $('job-form').elements.namedItem('refresh').disabled = kind !== 'results';
    const descriptions = { backfill: 'Collect historical odds in an explicit UTC interval. The worker splits fixture discovery into bounded windows and persists resumable progress.', upcoming: 'Discover upcoming fixtures and collect due current pre-match snapshots. Missing/inactive updates replace previous availability.', finished: 'Discover recently finished fixtures and acquire Winner histories within the configured finished-days window.', results: 'Fetch provider-native series results for pending finished fixtures. Map-only periods are not guessed into series labels.', import_existing: 'Import supported existing local archives and historical datasets into this database. No provider requests or arbitrary file paths.', remap: 'Rebuild conservative automatic candidates and links. Existing manual and rejected decisions remain protected. No provider requests.' };
    $('job-description').textContent = descriptions[kind]; $('submit-job').textContent = collection ? 'Review cost & queue job' : 'Queue local job';
  }
  $('job-kind').addEventListener('change', updateJobForm);
  $('job-form').addEventListener('submit', async (event) => {
    event.preventDefault(); const form = event.currentTarget; const data = Object.fromEntries(new FormData(form)); const kind = data.kind; const payload = {};
    try {
      if (kind === 'backfill') {
        if (!data.from || !data.to) throw new Error('Provide both UTC bounds.');
        const from = new Date(`${data.from}Z`); const to = new Date(`${data.to}Z`);
        if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime()) || from >= to) throw new Error('The UTC end must be later than the start.');
        payload.from = from.toISOString(); payload.to = to.toISOString();
      }
      if (kind === 'results' && data.refresh) payload.refresh = true;
      if (collectionKinds.has(kind)) {
        if (data.bookmakers?.trim()) payload.bookmakers = data.bookmakers.split(',').map((name) => name.trim()).filter(Boolean);
        ['billable_per_run', 'max_requests', 'max_history'].forEach((name) => { if (data[name]) { const value = Number(data[name]); if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer.`); payload[name] = value; } });
      }
    } catch (error) { message($('job-submit-state'), error.message, true); return; }
    await pending(form, $('job-submit-state'), async () => {
      if (collectionKinds.has(kind)) {
        if (!state.settings) state.settings = await api('/api/settings');
        const cap = payload.billable_per_run || state.settings?.billable_per_run;
        const scope = kind === 'backfill' ? ` UTC range ${timestamp(payload.from)} to ${timestamp(payload.to)}.` : '';
        if (!await confirmCost(`Queue ${kind} collection?${scope} Requested books: ${(payload.bookmakers || state.settings?.bookmakers || []).join(', ') || 'saved settings'}. This may attempt up to ${shown(cap)} billable calls per run, bounded by saved daily policy and provider quota. This is a real provider acquisition, not a dry run.`)) { message($('job-submit-state'), 'Not submitted.'); return; }
      }
      const result = await api('/api/jobs', 'POST', { kind, payload }); message($('job-submit-state'), `Queued job ${result.id}.`); await loadJobs();
    });
  });

  async function loadSettings() {
    message($('settings-state'), 'Loading saved settings…');
    try { populateSettings(await api('/api/settings')); message($('settings-state'), 'Saved policy loaded. Changes take effect only after saving.'); }
    catch (error) { message($('settings-state'), error.message, true); }
  }
  $('settings-form').addEventListener('submit', async (event) => {
    event.preventDefault(); const form = event.currentTarget; const data = Object.fromEntries(new FormData(form));
    const payload = { bookmakers: data.bookmakers.split(',').map((name) => name.trim()).filter(Boolean), scheduler_enabled: form.elements.namedItem('scheduler_enabled').checked };
    for (const [key, value] of Object.entries(data)) if (!['bookmakers', 'scheduler_enabled'].includes(key)) payload[key] = Number(value);
    if (!payload.bookmakers.length || new Set(payload.bookmakers).size !== payload.bookmakers.length) { message($('settings-save-state'), 'Enter at least one distinct bookmaker slug; duplicates are not allowed.', true); return; }
    await pending(form, $('settings-save-state'), async () => {
      if (payload.scheduler_enabled && !await confirmCost(`Save settings with scheduled collection enabled? The worker will be authorized to make future provider requests every ${payload.scheduler_interval_minutes} minutes, with local caps of ${payload.billable_per_run} billable attempts per run and ${payload.billable_per_day} per UTC day. This authorization remains active after closing the page.`)) { message($('settings-save-state'), 'Settings not saved.'); return; }
      populateSettings(await api('/api/settings', 'PUT', payload)); message($('settings-save-state'), 'Settings saved.'); await loadStatus();
    });
  });
  function lossTable(cohorts) {
    return tableWrap(['Bookmaker', 'Labeled matches', 'Natural-log loss'], Object.entries(cohorts || {}).map(([book, value]) => [book, shown(value.matches), number(value.log_loss, 6)]));
  }
  async function loadMetrics() {
    message($('metrics-state'), 'Calculating observed research cohorts…');
    try {
      const data = await api('/api/metrics'); const target = $('metrics-content'); clear(target);
      const own = section('Available / own cohorts'); own.append(el('p', 'Different match coverage means these values must not be used as a direct bookmaker ranking.', 'muted'));
      if (Object.keys(data.available_cohorts || {}).length) own.append(lossTable(data.available_cohorts)); else own.append(el('p', 'No available bookmaker cohorts yet.', 'empty-state')); target.append(own);
      const pairs = section('Pairwise common cohorts'); const rows = Object.entries(data.pairwise_common_cohorts || {});
      if (!rows.length) pairs.append(el('p', 'No pairwise cohorts available.', 'empty-state'));
      rows.forEach(([name, cohort]) => {
        const block = el('div', null, 'cohort-block'); block.append(el('h4', name.replaceAll('__', ' vs ')));
        block.append(el('p', `${shown(cohort.matches)} exact common labeled fixtures. Empty cohorts have no score.`));
        block.append(tableWrap(['Bookmaker', 'Log loss on this same cohort'], Object.entries(cohort.log_loss || {}).map(([book, loss]) => [book, number(loss, 6)])));
        if (cohort.fixture_ids?.length) block.append(jsonDetails('Exact cohort fixture IDs', cohort.fixture_ids)); pairs.append(block);
      }); target.append(pairs);
      const common = section('All requested books · common cohort'); const all = data.common_all_requested_books;
      if (!all) common.append(el('p', 'No common-all-books cohort available.', 'empty-state'));
      else { common.append(el('p', `${shown(all.matches)} exact common labeled fixtures.`)); common.append(tableWrap(['Bookmaker', 'Log loss on common cohort'], Object.entries(all.log_loss || {}).map(([book, loss]) => [book, number(loss, 6)]))); if (all.fixture_ids?.length) common.append(jsonDetails('Exact common fixture IDs', all.fixture_ids)); } target.append(common);
      const evidence = section('Composition, missingness & limits');
      evidence.append(facts({ mapped_fixture_count: data.mapped_fixture_count, uninformative_0_5_reference: data.uninformative_0_5_reference, convention: data.convention }));
      if (data.source_composition) evidence.append(jsonDetails('Observed label source composition', data.source_composition));
      if (data.limits) evidence.append(typeof data.limits === 'string' ? el('p', data.limits) : jsonDetails('Interpretation limits', data.limits));
      const other = Object.fromEntries(Object.entries(data).filter(([key]) => !['available_cohorts', 'pairwise_common_cohorts', 'common_all_requested_books', 'convention', 'uninformative_0_5_reference', 'limits', 'source_composition', 'mapped_fixture_count'].includes(key)));
      if (Object.keys(other).length) evidence.append(jsonDetails('Coverage and additional observed metrics', other));
      evidence.append(jsonDetails('Complete metrics response', data)); target.append(evidence); message($('metrics-state'), 'Metrics reflect the current persisted database, not model or betting performance.');
    } catch (error) { empty($('metrics-content'), 'Metrics could not be loaded. Refresh metrics to retry.'); message($('metrics-state'), error.message, true); }
  }
  const exports = { fixtures: 'Provider fixtures', history: 'Historical matches', quotes: 'All quote observations', mapped_quotes: 'Mapped research quotes', states: 'Full normalized odds states', mappings: 'Identity mappings', jobs: 'Job ledger' };
  Object.entries(exports).forEach(([name, title]) => { const anchor = link(`${title} · CSV`, `/api/export/${name}.csv`, true); anchor.className = 'export-link'; $('export-links').append(anchor); });

  function navigate() {
    const name = location.hash.slice(1); const page = pages.includes(name) ? name : 'dashboard';
    if (state.page === page) return; state.page = page;
    pages.forEach((entry) => { $(`page-${entry}`).hidden = entry !== page; });
    document.querySelectorAll('nav [data-page]').forEach((anchor) => { if (anchor.dataset.page === page) anchor.setAttribute('aria-current', 'page'); else anchor.removeAttribute('aria-current'); });
    $('global-error').hidden = true; document.title = `${document.querySelector(`#page-${page} h1`).textContent} · CS2 Research Desk`;
    if (page === 'dashboard') loadStatus();
    else if (page === 'fixtures' || page === 'mappings') loadFixtures(page);
    else if (page === 'history') loadHistory();
    else if (page === 'jobs') loadJobs();
    else if (page === 'settings') loadSettings();
    else if (page === 'research') loadMetrics();
  }
  [['fixture-filters', 'fixtures'], ['mapping-filters', 'mappings']].forEach(([id, mode]) => $(id).addEventListener('submit', (event) => { event.preventDefault(); state[mode].offset = 0; loadFixtures(mode); }));
  $('history-filters').addEventListener('submit', (event) => { event.preventDefault(); state.history.offset = 0; loadHistory(); });
  [['refresh-dashboard', loadStatus], ['refresh-jobs', loadJobs], ['refresh-metrics', loadMetrics]].forEach(([id, load]) => $(id).addEventListener('click', async () => {
    const node = $(id); node.disabled = true; try { await load(); } finally { node.disabled = false; }
  }));
  $('close-detail').addEventListener('click', () => $('fixture-dialog').close());
  $('fixture-dialog').addEventListener('close', () => { detailVersion++; state.odds.version++; state.detailId = null; });
  window.addEventListener('hashchange', navigate);
  document.addEventListener('visibilitychange', () => { if (document.hidden) { clearTimeout(state.poll); state.poll = null; } else if (state.jobs.some((job) => activeJobs.has(job.status))) loadJobs(); });
  window.addEventListener('unhandledrejection', (event) => { globalError(event.reason instanceof Error ? event.reason : new Error(String(event.reason))); });
  updateJobForm(); navigate();
  if (state.page !== 'dashboard') loadStatus();
  loadJobs();
})();
