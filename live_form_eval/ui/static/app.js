// The page of `lfe ui`: choose a form and an extractor or a run, play, and watch the form fill (docs/ui.md).
//
//   Live run       the server starts the extractor on the conversation; what happens arrives as events
//   Review a run   a run folder is loaded at once and replayed; nothing is called
//
// Either way the page holds a Track (track.js) and draws it at the position of the Playhead (playhead.js), frame
// by frame: the conversation (talk.js), the form (form.js) and the small graphs of the score so far (charts.js).

import { $, clamp, clock, el, figure, follow, getJSON, pause, postJSON } from './util.js';
import { Playhead } from './playhead.js';
import { Popover } from './popover.js';
import { Trend } from './charts.js';
import { Track } from './track.js';
import { TalkView } from './talk.js';
import { FormView } from './form.js';

const ICONS = { play: 'M4 2.5v11l9-5.5z', pause: 'M4 3h3v10H4zM9 3h3v10H9z', stop: 'M3.5 3.5h9v9h-9z' };
const TILES = [
  { key: 'score', name: 'Score', hint: 'harmonic mean of the next four', lead: true },
  { key: 'completeness', name: 'Completeness', hint: 'was each value shown?' },
  { key: 'precision', name: 'Precision', hint: 'was the screen right?' },
  { key: 'speed', name: 'Speed', hint: 'how soon after it was said?' },
  { key: 'outcome', name: 'Outcome', hint: 'is the form right now?' },
  { key: 'price', name: 'Price', hint: 'no model call so far' },
];
const SCORE_NOTE = $('metrics-note').textContent;
const LOG_LINES = 2000;
const TABLE_ROWS = 150;

const state = {
  catalog: null,
  mode: 'live',       // 'live' or 'review'
  form: '',           // "dataset|split|conversation id"
  extractor: '',
  run: '',            // the run that is reviewed; '' = the conversation alone
  speed: 1,
  sound: false,
  track: null,
  phase: 'loading',   // loading, ready, starting, preparing, running, stopping, ended
  feed: null,
  turn: 0,            // grows with every choice: an answer that arrives for an earlier choice is dropped
  at: -1,
  stale: true,        // something to draw has changed
  redraw: true,       // the graphs' lines have changed
  hover: null,
  wasPlaying: false,
  ended: null,        // the run folder of the live run that just ended
};
let playhead, popover, talk, form;

// ---- words in the bar under the controls

function notice(text, { problem = false, action = null } = {}) {
  const node = $('notice');
  node.classList.toggle('problem', problem);
  node.replaceChildren(text);
  if (action) {
    const button = el('button', { type: 'button', class: 'link' }, action.label);
    button.addEventListener('click', action.run);
    node.append(button);
  }
}

function describeChoice() {
  const track = state.track;
  if (!track) return;
  if (state.mode === 'live') {
    if (state.phase === 'stopping') {
      notice('Stopping the run: its extractor’s close() ends what it opened for this conversation, then the run folder is written.');
    } else if (state.phase === 'running' || state.phase === 'preparing' || state.phase === 'starting') {
      notice(`${state.phase === 'running' ? 'Running' : 'Starting'} ${state.extractor} at ${state.speed}×.`
        + (state.speed === 1 ? '' : ' At another speed than 1×, the extractor’s delays are not what real time would give.'));
    } else if (state.phase === 'ended' && track.status) {
      const final = track.final;
      const how = track.status.status === 'ok' ? 'The run is over.' : `The run ended: ${track.reason || track.status.status}.`;
      const numbers = final ? ` Score ${figure(final.score)} · price €${figure(final.eur, 4)} · ${final.correct} correct, `
        + `${final.wrong} wrong, ${final.missing} missing, ${final.invented} invented.` : '';
      notice(`${how}${numbers}${state.ended ? ` Saved as runs/${state.ended}. ` : ' '}`, state.ended ? {
        // replayed in real time, whatever speed the run was made at
        action: { label: 'Review it', run: () => { state.mode = 'review'; state.run = state.ended; state.speed = 1; fillSpeeds(); choose(); } },
      } : {});
    } else {
      notice('A live run really runs the extractor: whatever it calls is called, and billed if it bills. '
        + 'A run from this page is one conversation, so it is never an official run.');
    }
  } else if (track.run) {
    const made = track.run.started_at ? new Date(track.run.started_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '';
    const status = track.status && track.status.status !== 'ok' ? ` · it ended: ${track.status.status}` : '';
    notice(`${track.run.name} · ${made} · made at ${track.run.speed}×${status}. Replayed from its run folder: nothing is called.`);
  } else {
    notice('The conversation alone: what the extractor would receive, and what the form should show. Choose a run to see one fill it.');
  }
}

// ---- the choices

const formOf = () => { const [dataset, split, id] = state.form.split('|'); return { dataset, split, id }; };
const runsOn = (key) => {
  const { dataset, split, id } = (([d, s, i]) => ({ dataset: d, split: s, id: i }))(key.split('|'));
  return state.catalog.runs.filter((run) => run.dataset === dataset && run.split === split
    && run.conversations.some(([conversation]) => conversation === id));
};

function options(select, groups, chosen) {
  const built = groups.map(({ label, items }) => {
    const made = items.map(({ value, text, title, disabled }) => el('option', { value, title, disabled, selected: value === chosen }, text));
    return label ? el('optgroup', { label }, made) : made;
  });
  select.replaceChildren(...built.flat());
}

function fillForms() {
  const names = state.catalog.settings.datasets;
  const groups = state.catalog.datasets.map((data) => ({
    label: `${names[data.name] || data.name} (${data.split})`,
    items: data.forms.map((item) => ({
      value: `${data.name}|${data.split}|${item.id}`,
      text: `${item.id} · ${clock(item.seconds)} · ${item.said} of ${item.fields} fields`,
    })),
  }));
  const sample = new Set(groups.flatMap((group) => group.items.map((item) => item.value)));
  if (state.mode === 'review') {  // a run may have been made on other conversations than the sample's
    const others = new Map();
    for (const run of state.catalog.runs) {
      for (const [conversation] of run.conversations) {
        const key = `${run.dataset}|${run.split}|${conversation}`;
        if (!sample.has(key)) others.set(key, `${conversation} · ${names[run.dataset] || run.dataset} (${run.split})`);
      }
    }
    if (others.size) groups.push({ label: 'In earlier runs, not in the sample', items: [...others].sort().map(([value, text]) => ({ value, text })) });
  }
  const known = groups.flatMap((group) => group.items.map((item) => item.value));
  if (!known.includes(state.form)) state.form = known[0] || '';
  options($('form'), groups, state.form);
}

function fillSource() {
  const select = $('source');
  if (state.mode === 'live') {
    const found = state.catalog.extractors;
    const usable = found.filter((entry) => !entry.error);
    if (!usable.some((entry) => entry.id === state.extractor)) state.extractor = usable.length ? usable[0].id : '';
    const items = found.map((entry) => (entry.error
      ? { value: `!${entry.file}`, text: `${entry.file}: ${entry.error}`, disabled: true }
      : { value: entry.id, text: `${entry.name} — ${entry.file}`, title: entry.about }));
    options(select, [{ items: items.length ? items
      : [{ value: '', text: `No extractor found in ${state.catalog.settings.folder}`, disabled: true }] }], state.extractor);
  } else {
    const runs = runsOn(state.form);
    if (!runs.some((run) => run.id === state.run)) state.run = runs.length ? runs[0].id : '';
    const id = formOf().id;
    const items = runs.map((run) => {
      const made = run.started_at ? new Date(run.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
      const ended = (run.conversations.find(([conversation]) => conversation === id) || [])[1];
      return { value: run.id, text: `${run.name} · ${made} · ${run.speed}×${run.official ? ' · official' : ''}${ended && ended !== 'ok' ? ` · ${ended}` : ''}`,
               title: run.id };
    });
    options(select, [{ items: [...items, { value: '', text: items.length ? 'No run: the conversation alone' : 'No run on this form yet: the conversation alone' }] }], state.run);
  }
}

function fillSpeeds() {
  const speeds = state.catalog.settings.speeds;
  if (!speeds.includes(state.speed)) state.speed = 1;
  options($('speed'), [{ items: speeds.map((speed) => ({ value: String(speed), text: `${speed}×` })) }], String(state.speed));
}

function address() {
  const query = new URLSearchParams({ mode: state.mode, form: state.form, speed: state.speed, sound: state.sound ? '1' : '0',
                                      view: talk.view, fields: form.everyField ? 'all' : 'play' });
  if (state.extractor) query.set('extractor', state.extractor);
  if (state.run) query.set('run', state.run);
  history.replaceState(null, '', `?${query}`);
}

function controls() {
  const live = state.mode === 'live';
  const busy = ['starting', 'preparing', 'running', 'stopping'].includes(state.phase);
  $('mode-live').setAttribute('aria-pressed', String(live));
  $('mode-review').setAttribute('aria-pressed', String(!live));
  for (const id of ['mode-live', 'mode-review', 'form', 'source', 'speed']) $(id).disabled = busy;
  $('speed').disabled = busy && live;
  $('source-label').textContent = live ? 'Extractor' : 'Run';
  const play = $('play');
  let label, icon;
  if (live) {
    label = !busy ? 'Start run' : state.phase === 'running' ? 'Stop run' : state.phase === 'stopping' ? 'Stopping…' : 'Starting…';
    icon = busy ? ICONS.stop : ICONS.play;
    // a run that is stopping can't be stopped again: its extractor is closing what it opened
    play.disabled = !state.track || state.phase === 'loading' || state.phase === 'stopping' || (!busy && !state.extractor);
  } else {
    label = playhead.playing ? 'Pause' : 'Play';
    icon = playhead.playing ? ICONS.pause : ICONS.play;
    play.disabled = !state.track || state.phase === 'loading';
  }
  $('play-label').textContent = label;
  $('play-icon').setAttribute('d', icon);
  $('scrub').disabled = !state.track || playhead.live || busy;
  const tooFast = state.speed > state.catalog.settings.audio_max_speed;
  const sound = $('sound');
  sound.setAttribute('aria-pressed', String(state.sound));
  $('sound-label').textContent = state.sound ? (tooFast ? 'Sound on (too fast)' : 'Sound on') : 'Sound off';
  $('sound-waves').style.display = state.sound ? '' : 'none';
  sound.title = tooFast ? `The recording is played up to ${state.catalog.settings.audio_max_speed}×: faster, nobody can follow it.`
    : 'Play the recording along with the conversation';
}

// ---- loading what was chosen

function stopFeed() {
  if (state.feed) state.feed.close();
  state.feed = null;
}

function show(track) {
  state.track = track;
  state.ended = null;
  playhead.reset(track.end);
  talk.load(track);
  form.load(track);
  for (const tile of TILES) tile.trend.setEnd(track.end);
  $('scrub').max = track.end;
  $('length').textContent = clock(track.end);
  $('output').hidden = true;
  $('output-lines').textContent = '';
  $('metrics-note').textContent = SCORE_NOTE;
  state.stale = true;
  state.redraw = true;
  sound();
}

function lost() {
  notice('The page lost its server. Start `lfe ui` again and reload the page.', { problem: true });
}

/** Load what the selectors say: a run of the form to review, or the form alone (also before a live run). */
async function choose() {
  const turn = state.turn += 1;
  stopFeed();
  fillForms();
  fillSource();
  state.phase = 'loading';
  controls();
  address();
  if (!state.form) {
    notice(state.catalog.problems.join(' ') || 'There is no form to show.', { problem: true });
    return;
  }
  $('main').classList.add('loading');
  const { dataset, split, id } = formOf();
  try {
    if (state.mode === 'review' && state.run) {
      const data = await getJSON(`/api/review?run=${encodeURIComponent(state.run)}&id=${encodeURIComponent(id)}`);
      if (turn !== state.turn) return;
      const track = new Track('review', data.conversation, Math.max(data.end, data.conversation.input_end));
      track.load(data.updates);
      data.calls.forEach((call) => track.call(call));
      track.setWalks(data.fields);
      Object.assign(track, { final: data.final, status: data.status, run: data.run });
      show(track);
      state.feed = follow(data.feed, (event) => { if (turn === state.turn) reviewed(event); }, lost);
    } else {
      const conversation = await getJSON(`/api/form?dataset=${dataset}&split=${split}&id=${encodeURIComponent(id)}`);
      if (turn !== state.turn) return;
      show(new Track(state.mode === 'live' ? 'live' : 'alone', conversation,
        state.mode === 'live' ? conversation.end : conversation.input_end));
    }
    state.phase = 'ready';
    describeChoice();
  } catch (error) {
    if (turn !== state.turn) return;
    state.phase = 'ready';
    notice(`This could not be loaded: ${error.message}`, { problem: true });
  } finally {
    if (turn === state.turn) {
      $('main').classList.remove('loading');
      controls();
    }
  }
}

/** The score over a reviewed run arrives in pieces while the server computes it. */
function reviewed(event) {
  const track = state.track;
  if (event.event === 'plan') track.planned = event.moments;
  if (event.event === 'points') {
    track.series.push(...event.points);
    state.redraw = true;
    if (track.planned) {
      $('metrics-note').textContent = `Computing the score over time: ${track.series.length} of ${track.planned} moments. `
        + 'Notes that nobody judged before are read by a language model first, which takes a moment.';
    }
  }
  if (event.event === 'problem') notice(event.message, { problem: true });
  if (event.event === 'done') {
    track.scored = true;
    $('metrics-note').textContent = SCORE_NOTE;
  }
  state.stale = true;
}

// ---- a live run

async function startLive() {
  const turn = state.turn += 1;
  stopFeed();
  state.phase = 'starting';
  controls();
  describeChoice();
  const conversation = state.track.conversation;
  try {
    const run = await postJSON('/api/live', { ...formOf(), extractor: state.extractor, speed: state.speed });
    if (turn !== state.turn) return;
    show(new Track('live', conversation, conversation.end));
    state.phase = 'preparing';
    state.feed = follow(run.feed, (event) => { if (turn === state.turn) happened(event); }, lost);
  } catch (error) {
    state.phase = 'ready';
    notice(`The run could not start: ${error.message}`, { problem: true });
  }
  controls();
}

/** Stop the live run. Its process ends the extractor's run() and then lets its close() finish (it ends what the
 *  extractor opened for the conversation), so the run is over a moment later: when its `ended` event arrives. */
function stopLive() {
  const before = state.phase;
  state.phase = 'stopping';
  describeChoice();
  postJSON('/api/live/stop').catch((error) => {
    if (state.phase === 'stopping') state.phase = before;  // it goes on: it can be stopped again
    controls();
    notice(error.message, { problem: true });
  });
}

function logged(line) {
  const track = state.track;
  if (track.log.length >= LOG_LINES) return;
  track.log.push(line);
  $('output').hidden = false;
  $('output-lines').textContent += `${line}\n`;
  $('output-toggle').textContent = `Extractor output (${track.log.length} line${track.log.length === 1 ? '' : 's'})`;
}

function openOutput(open) {
  $('output-lines').hidden = !open;
  $('output-toggle').setAttribute('aria-expanded', String(open));
}

/** One event of a live run (ui/worker.py lists them; the server adds `score`, `log` and `problem`). */
function happened(event) {
  const track = state.track;
  switch (event.event) {
    case 'started':
      playhead.startLive(event.speed);
      if (state.phase !== 'stopping') state.phase = 'running';
      break;
    case 'tick':
      playhead.follow(event.t);
      break;
    case 'update':
      track.change(event.t, event.field, event.value);
      break;
    case 'call':  // a use, as a review has it (the server reads it like the evaluation: ui/playback.py, calls_payload)
      track.call(event);
      state.redraw = true;
      break;
    case 'score':
      track.series.push(...event.points);
      track.setWalks(event.fields);
      if (event.final) track.final = event.final;
      state.redraw = true;
      break;
    case 'log':
      logged(event.line);
      break;
    case 'problem':
      notice(event.message, { problem: true });
      break;
    case 'failed':
      event.error.split('\n').forEach(logged);
      openOutput(true);
      track.failed = true;
      state.phase = 'ready';
      notice('The run could not start. The extractor’s output below says why.', { problem: true });
      break;
    case 'ended':
      if (track.failed) return;  // it never started: what is on the page already says why
      track.status = event.status;
      track.reason = event.reason;
      state.ended = event.run;
      playhead.endLive(event.status.run_end ?? playhead.now());
      state.phase = 'ended';
      if (event.status.error) { String(event.status.error).split('\n').forEach(logged); }
      break;
    case 'done':
      refresh();  // the run that was just written joins the list of runs
      break;
    default:
  }
  if (['started', 'ended', 'failed'].includes(event.event)) { controls(); if (event.event !== 'failed') describeChoice(); }
  if (event.event === 'score' && state.phase === 'ended') describeChoice();
  state.stale = true;
}

// ---- the recording

async function sound() {
  const audio = $('audio'), track = state.track, turn = state.turn;
  audio.pause();
  audio.removeAttribute('src');
  playhead.setSound(false);
  if (!state.sound || !track) { controls(); return; }
  const { dataset, split, id } = track.conversation;
  try {
    let recording = track.conversation.audio;
    if (recording.state !== 'ready') {
      recording = await postJSON('/api/audio/fetch', { dataset, split, id });
      while (recording.state === 'fetching' && turn === state.turn && state.sound) {
        const share = recording.total ? ` ${Math.round(100 * recording.done / recording.total)}%` : '';
        $('sound-label').textContent = `Fetching the recording…${share}`;
        await pause(400);
        recording = await getJSON(`/api/audio/state?dataset=${dataset}&id=${encodeURIComponent(id)}`);
      }
    }
    if (turn !== state.turn || !state.sound) return;
    if (recording.state !== 'ready') throw new Error(recording.message || 'the recording is not available');
    track.conversation.audio = recording;
    audio.src = `/api/audio?dataset=${dataset}&id=${encodeURIComponent(id)}`;
    playhead.setSound(true);
  } catch (error) {
    state.sound = false;
    notice(`No sound for this conversation: ${error.message}`, { problem: true });
  }
  controls();
  address();
}

// ---- the score so far

function makeTiles() {
  for (const tile of TILES) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${tile.name} so far, over the conversation. The table has every value.`);
    tile.figure = el('div', { class: 'figure' }, '–');
    tile.note = el('div', { class: 'hint' }, tile.hint);
    $('tiles').append(el('div', { class: `tile${tile.lead ? ' lead' : ''}` }, el('div', { class: 'name' }, tile.name), tile.figure, tile.note, svg));
    tile.trend = tile.key === 'price'
      ? new Trend(svg, { steps: true, top: (max) => `€${max < 0.01 ? max.toFixed(4) : max.toFixed(2)}` })
      : new Trend(svg, { max: 1, top: () => '1' });
    svg.addEventListener('pointermove', (event) => hover(tile.trend.timeAt(event.clientX), event));
    svg.addEventListener('pointerleave', () => hover(null));
    svg.addEventListener('click', (event) => seek(tile.trend.timeAt(event.clientX)));
  }
}

const euros = (value) => `€${value.toFixed(value < 0.01 ? 6 : value < 0.1 ? 4 : 2)}`;
const UNIT_NAMES = { cache_read_tokens: 'Read from cache', input_tokens: 'Fresh input', cache_write_tokens: 'Written to cache',
                     output_tokens: 'Output', '(not split)': 'Not split by unit' };  // as the report names them
const unitName = (unit) => UNIT_NAMES[unit] || unit.replaceAll('_', ' ');

/** "57 model calls so far · 3 other uses": what the price so far is made of. */
function usesSoFar(track, t) {
  const calls = track.callsUntil(t), other = track.otherUntil(t);
  return `${calls ? `${calls} model call${calls === 1 ? '' : 's'}` : 'no model call'} so far`
    + (other ? ` · ${other} other use${other === 1 ? '' : 's'}` : '');
}

function drawTrends(t) {
  const track = state.track;
  const known = track.series.length ? track.series[track.series.length - 1].t : 0;
  for (const tile of TILES) {
    if (tile.key === 'price') {
      const points = [{ t: 0, v: 0 }, ...track.calls.map((call, k) => ({ t: call.t, v: track.spent[k] }))];
      tile.trend.set(points, track.kind === 'live' && !track.status ? t : track.end, track.calls.map((call) => call.t));
    } else {
      tile.trend.set(track.series.map((point) => ({ t: point.t, v: point[tile.key] })), known);
    }
  }
}

function measures(t) {
  const track = state.track, point = track.scoreAt(t);
  const waiting = !track.scored && (!point || (track.series.length && track.series[track.series.length - 1].t < t - 1));
  return TILES.map((tile) => {
    if (tile.key === 'price') return { tile, text: euros(track.priceAt(t)) };
    return { tile, text: waiting ? '…' : figure(point ? point[tile.key] : null) };
  });
}

function drawMeasures(t) {
  const track = state.track;
  for (const { tile, text } of measures(t)) {
    tile.figure.textContent = text;
    tile.trend.setNow(t);
    if (tile.key === 'price') tile.note.textContent = usesSoFar(track, t);
  }
}

function hover(t, event) {
  const readout = $('readout');
  state.hover = t;
  for (const tile of TILES) tile.trend.setHover(t);
  if (t === null || !state.track) { readout.hidden = true; return; }
  const track = state.track, other = track.otherUntil(t);
  readout.replaceChildren(el('table', {}, el('caption', {}, `At ${clock(t, true)}`),
    measures(t).map(({ tile, text }) => el('tr', {}, el('th', {}, tile.name), el('td', {}, text))),
    // the price so far, unit by unit (it comes right after the price, the last of the measures)
    track.unitsAt(t).map(([unit, eur]) => el('tr', { class: 'part' }, el('th', {}, unitName(unit)), el('td', {}, euros(eur)))),
    el('tr', {}, el('th', {}, 'Model calls'), el('td', {}, String(track.callsUntil(t)))),
    other ? el('tr', {}, el('th', {}, 'Other uses'), el('td', {}, String(other))) : null));
  readout.hidden = false;
  readout.style.left = `${clamp(event.clientX + 14, 8, window.innerWidth - readout.offsetWidth - 8)}px`;
  readout.style.top = `${clamp(event.clientY + 16, 8, window.innerHeight - readout.offsetHeight - 8)}px`;
}

function numbers() {
  const track = state.track;
  if (!track) return;
  const every = Math.max(1, Math.ceil(track.series.length / TABLE_ROWS));
  const kept = track.series.filter((_, k) => k % every === 0 || k === track.series.length - 1);
  const head = ['Time', 'Score', 'Completeness', 'Precision', 'Speed', 'Outcome', 'Price (€)', 'Model calls', 'Other uses'];
  $('numbers-body').replaceChildren(kept.length ? el('table', { class: 'plain' },
    el('thead', {}, el('tr', {}, head.map((name) => el('th', {}, name)))),
    el('tbody', {}, kept.map((point) => el('tr', {},
      [clock(point.t, true), figure(point.score), figure(point.completeness), figure(point.precision), figure(point.speed),
       figure(point.outcome), figure(track.priceAt(point.t), 4), String(track.callsUntil(point.t)),
       String(track.otherUntil(point.t))]
        .map((cell) => el('td', { class: 'n' }, cell))))))
    : el('p', {}, 'There is no score yet: nothing was said for the form so far, or no run is chosen.'));
  $('numbers').showModal();
}

// ---- the position

function seek(t) {
  if (!state.track || playhead.live) return;
  playhead.seek(t);
  state.stale = true;
}

function frame() {
  const track = state.track;
  if (track) {
    const t = playhead.now();
    if (playhead.playing !== state.wasPlaying) {
      state.wasPlaying = playhead.playing;
      controls();
    }
    if (state.redraw) {
      state.redraw = false;
      state.stale = true;
      drawTrends(t);
    }
    if (t !== state.at || state.stale) {
      state.at = t;
      state.stale = false;
      talk.render(t);
      form.render(t, playhead.playing ? playhead.rate : 0);
      drawMeasures(t);
      $('clock').textContent = clock(t);
      if (document.activeElement !== $('scrub')) $('scrub').value = t;
    }
  }
  requestAnimationFrame(frame);
}

// ---- keeping the lists fresh: a new extractor file or a new run shows up by itself

async function refresh() {
  let catalog;
  try {
    catalog = await getJSON('/api/state');
  } catch {
    return;  // the server is away; the next round tries again
  }
  const before = state.catalog;
  state.catalog = catalog;
  const ids = (found) => JSON.stringify(found.extractors.map((entry) => [entry.id, entry.error, entry.about]));
  const runs = (found) => JSON.stringify(found.runs.map((run) => run.id));
  const busy = ['starting', 'preparing', 'running', 'stopping', 'loading'].includes(state.phase);
  if (ids(before) !== ids(catalog) && state.mode === 'live') {
    const known = new Set(before.extractors.map((entry) => entry.id));
    const fresh = catalog.extractors.filter((entry) => !entry.error && !known.has(entry.id));
    if (!busy) { fillSource(); controls(); }
    if (fresh.length && !busy) notice(`Found ${fresh.length === 1 ? 'a new extractor' : 'new extractors'}: ${fresh.map((entry) => entry.id).join(', ')}.`);
  }
  if (runs(before) !== runs(catalog) && state.mode === 'review' && !busy) {
    const chosen = state.run;
    fillForms();
    fillSource();
    if (state.run !== chosen) choose();  // the run that was shown is gone
  }
}

// ---- start

function colours({ chrome, status, series }) {
  const set = (name, value) => document.documentElement.style.setProperty(name, value);
  for (const name of ['surface', 'page', 'primary', 'secondary', 'muted', 'grid', 'axis']) set(`--${name}`, chrome[name]);
  set('--series', series[0]);
  set('--right', status.correct);
  set('--waiting', status.missing);
  set('--invented', status.invented);
  set('--wrong', status.wrong);
}

function divider() {
  const work = $('work'), handle = $('split');
  const apply = (share) => {
    const kept = clamp(share, 22, 70);
    work.style.setProperty('--talk', `${kept}%`);
    handle.setAttribute('aria-valuenow', String(Math.round(kept)));
    localStorage.setItem('lfe-ui-talk', String(kept));
  };
  apply(Number(localStorage.getItem('lfe-ui-talk')) || 40);
  handle.addEventListener('pointerdown', (down) => {
    handle.setPointerCapture(down.pointerId);
    const move = (event) => { const box = work.getBoundingClientRect(); apply(100 * (event.clientX - box.left) / box.width); };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', () => handle.removeEventListener('pointermove', move), { once: true });
  });
  handle.addEventListener('keydown', (event) => {
    const now = Number(handle.getAttribute('aria-valuenow'));
    if (event.key === 'ArrowLeft') apply(now - 2);
    if (event.key === 'ArrowRight') apply(now + 2);
  });
}

function pressed(on, off, isOn) {
  $(on).setAttribute('aria-pressed', String(isOn));
  $(off).setAttribute('aria-pressed', String(!isOn));
}

async function start() {
  popover = new Popover($('pop'));
  playhead = new Playhead($('audio'));
  talk = new TalkView({ caption: $('caption'), said: $('said'), scroller: $('scroller'), rows: $('rows'), follow: $('follow') }, popover, seek);
  form = new FormView({ cards: $('cards'), note: $('form-note'), legend: $('legend'), drawer: $('drawer'),
                        drawerTitle: $('drawer-title'), drawerBody: $('drawer-body') }, popover);
  makeTiles();
  divider();

  try {
    state.catalog = await getJSON('/api/state');
  } catch (error) {
    notice(`The page could not reach its server: ${error.message}`, { problem: true });
    return;
  }
  colours(state.catalog.colors);
  playhead.soundUpTo = state.catalog.settings.audio_max_speed;

  const asked = new URLSearchParams(location.search);
  state.mode = asked.get('mode') === 'review' ? 'review' : 'live';
  state.form = asked.get('form') || '';
  state.extractor = asked.get('extractor') || '';
  state.run = asked.get('run') || '';
  state.speed = Number(asked.get('speed')) || 1;
  state.sound = asked.get('sound') === '1';
  talk.view = asked.get('view') === 'messages' ? 'messages' : 'turns';
  form.everyField = asked.get('fields') === 'all';
  pressed('view-turns', 'view-messages', talk.view === 'turns');
  pressed('show-play', 'show-all', !form.everyField);
  fillSpeeds();

  $('mode-live').addEventListener('click', () => { if (state.mode !== 'live') { state.mode = 'live'; choose(); } });
  $('mode-review').addEventListener('click', () => { if (state.mode !== 'review') { state.mode = 'review'; choose(); } });
  $('form').addEventListener('change', (event) => { state.form = event.target.value; choose(); });
  $('source').addEventListener('change', (event) => {
    if (state.mode === 'live') { state.extractor = event.target.value; address(); controls(); } else { state.run = event.target.value; choose(); }
  });
  $('speed').addEventListener('change', (event) => {
    state.speed = Number(event.target.value);
    if (!playhead.live) playhead.setRate(state.speed);
    controls();
    address();
  });
  $('play').addEventListener('click', () => {
    if (state.mode === 'live') {
      if (['preparing', 'running'].includes(state.phase)) stopLive();
      else if (state.phase !== 'starting' && state.phase !== 'stopping') startLive();
    } else if (playhead.playing) {
      playhead.pause();
    } else {
      playhead.setRate(state.speed);
      playhead.play();
    }
    controls();
  });
  $('sound').addEventListener('click', () => { state.sound = !state.sound; sound(); address(); });
  $('scrub').addEventListener('input', (event) => seek(Number(event.target.value)));
  $('scrub').addEventListener('change', (event) => event.target.blur());
  $('view-turns').addEventListener('click', () => { talk.setView('turns'); pressed('view-turns', 'view-messages', true); state.stale = true; address(); });
  $('view-messages').addEventListener('click', () => { talk.setView('messages'); pressed('view-turns', 'view-messages', false); state.stale = true; address(); });
  $('show-play').addEventListener('click', () => { form.showEvery(false); pressed('show-play', 'show-all', true); state.stale = true; address(); });
  $('show-all').addEventListener('click', () => { form.showEvery(true); pressed('show-play', 'show-all', false); state.stale = true; address(); });
  $('drawer-close').addEventListener('click', () => form.close());
  $('numbers-open').addEventListener('click', numbers);
  $('numbers-close').addEventListener('click', () => $('numbers').close());
  $('output-toggle').addEventListener('click', () => openOutput($('output-lines').hidden));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') form.close();
    const typing = event.target instanceof Element && event.target.closest('select, input, button, [role="button"], dialog');
    if (typing || state.mode !== 'review' || !state.track) return;
    if (event.key === ' ') { event.preventDefault(); $('play').click(); }
    if (event.key === 'ArrowLeft') seek(playhead.now() - 5);
    if (event.key === 'ArrowRight') seek(playhead.now() + 5);
  });

  await choose();
  setInterval(refresh, state.catalog.settings.refresh_s * 1000);
  requestAnimationFrame(frame);
}

start();
