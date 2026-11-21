// How a field's state is put into words and marks, and what the windows on a value hold. Shared by the form's
// cards and by the conversation's timeline, so that both say the same thing in the same words.

import { clock, el, figure, lastAt, plain } from './util.js';

const SAME_MOMENT = 1e-6;

/** The marks of a field: a sign and a word (a colour is never the only way to tell them apart). */
export const MARKS = {
  right: { sign: '✓', word: 'Right', means: 'it shows the value that was said' },
  close: { sign: '≈', word: 'Close', means: 'it shows something close to it' },
  wrong: { sign: '✕', word: 'Wrong', means: 'it shows another value' },
  waiting: { sign: '…', word: 'Waiting', means: 'a value was said and is not on the form yet' },
  invented: { sign: '+', word: 'Invented', means: 'it shows a value where the field should be empty' },
  unread: { sign: '?', word: 'Not graded', means: '' },
  empty: { sign: '', word: 'Empty', means: '' },
  idle: { sign: '', word: '', means: '' },
};
export const LEGEND = ['right', 'close', 'wrong', 'waiting', 'invented'];

const SOURCES = {
  span: "the dataset's own annotation of the words",
  label: 'the words two language models quoted when they corrected the dataset’s label',
  rule: "a rule on the caller's turn",
  llm: 'the words a language model pointed at',
  turn: "the end of the caller's turn",
  end: 'no words say it: the end of the conversation',
};
const COUNTS = {
  landing: 'the value that was said',
  repair: 'back on the value that was said',
};

export function mark(kind) {
  const { sign, word } = MARKS[kind] || MARKS.idle;
  return el('span', { class: `mark ${kind}`, role: 'img', 'aria-label': word || 'nothing yet' }, sign);
}

/** How a walk's display is graded, as a mark kind. */
export const gradeKind = (display) => (!display ? 'unread' : display.grade >= 1 ? 'right' : display.grade > 0 ? 'close' : 'wrong');

/** A value in full: a list item by item, a text with its line breaks, nothing as a word. */
export function valueNode(value, nothing = 'nothing') {
  if (value === null || value === undefined) return el('span', { class: 'type' }, nothing);
  if (Array.isArray(value)) return el('ul', { class: 'units' }, value.map((item) => el('li', {}, '•', el('span', {}, String(item)))));
  return document.createTextNode(plain(value));
}

/** What a gold step adds to the one before it, and what it ends: for a field that grows piece by piece (the
 *  options of a multiselect, the facts of a note). None for a field that holds one value. */
export function stepChange(field, index) {
  const now = field.steps[index].value, before = index > 0 ? field.steps[index - 1].value : null;
  if (!Array.isArray(now) && !Array.isArray(before)) return null;
  const was = Array.isArray(before) ? before : [], is = Array.isArray(now) ? now : [];
  return { added: is.filter((piece) => !was.includes(piece)), ended: was.filter((piece) => !is.includes(piece)) };
}

/** A gold step as one short line: its value, or what it adds. */
export function stepLine(field, index) {
  const change = stepChange(field, index);
  if (!change) return field.steps[index].value === null ? 'withdrawn' : plain(field.steps[index].value);
  const more = (pieces) => (pieces.length > 1 ? ` (and ${pieces.length - 1} more)` : '');
  if (change.added.length) return `+ ${change.added[0]}${more(change.added)}`;
  return change.ended.length ? `− ${change.ended[0]}${more(change.ended)}` : 'unchanged';
}

function rows(pairs) {
  return el('dl', {}, pairs.filter(Boolean).map(([name, value]) => [el('dt', {}, name), el('dd', {}, value)]));
}

/** What the field asks for: its prompt, the line an extractor gets for it. It is what says whether a value
 *  belongs in the field, so every window on a value starts with it. */
const asks = (field) => ['It asks for', field.prompt || '—'];

/** The gold step of a field that holds at second `time` (index -1: none has been said yet), and whether the field
 *  should be empty then: nothing was said for it so far, or what was said was withdrawn. */
export function goldAt(field, time) {
  const index = lastAt(field.steps, time, (step) => step.said);
  return { index, empty: index < 0 || field.steps[index].value === null };
}

/** Why a field should be empty while its gold step `index` holds (-1: none has been said), as a sentence. */
function shouldBeEmpty(field, index) {
  if (index >= 0) return `The value of this field was withdrawn at ${clock(field.steps[index].said, true)}, so it should be empty.`;
  if (field.steps.length) {
    return `No value had been said for this field by then (the first comes at ${clock(field.steps[0].said, true)}), `
      + 'so it should still be empty.';
  }
  return 'The labels of this conversation hold no value for this field, so it should stay empty.';
}

/** The judge's reading of one text against the facts of one gold step: which facts the text says, and which of
 *  its sentences or items the facts support. Null when that text was not judged against that step. */
export function judgedAgainst(track, field, time, stepIndex) {
  const walk = track.walks[field.id];
  const judged = walk && walk.judged && walk.judged.find((item) => Math.abs(item.t - time) < SAME_MOMENT);
  const against = judged && judged.against[String(stepIndex)];
  return against ? { units: judged.units, unitOk: against.units, factOk: against.facts } : null;
}

/** The window on a gold step: what the field should show from when, where that comes from, what the form did. */
export function goldWindow(track, field, index) {
  const step = field.steps[index], next = field.steps[index + 1];
  const walk = track.walks[field.id];
  const read = walk && walk.steps[index];
  const change = stepChange(field, index);
  let form = null;
  if (read && read.shown === null) form = track.kind === 'live' ? 'has not shown it so far' : 'never showed it';
  else if (read && read.landed) form = `showed it at ${clock(read.shown, true)}, ${figure(Math.max(0, read.delay), 1)} s after it was said`;
  else if (read) form = `came closest at ${clock(read.shown, true)} (score ${figure(read.score, 2)})`;
  return rows([
    ['Field', field.label],
    asks(field),
    ['Should show', valueNode(step.value, 'nothing: the value is withdrawn')],
    change && change.added.length && ['New here', valueNode(change.added)],
    change && change.ended.length && ['No longer', valueNode(change.ended)],
    ['From', `${clock(step.said, true)}, when the extractor got the words`
      + (step.spoken !== null ? ` (heard at ${clock(step.spoken, true)} in the recording)` : '')],
    ['Until', next ? clock(next.said, true) : 'the end'],
    step.heard && ['Heard', `“… ${step.heard}”`],
    step.source && ['Its time comes from', SOURCES[step.source] || step.source],
    form && ['The form', form],
  ]);
}

/** How the evaluation read a value the form showed at second `time`: its grade as a mark, and two special cases.
 *
 *  `early`: a value that is shown before the words that say it have arrived is graded against what had been said
 *  until then; from the moment it is said, it counts as right, with no delay (timeline/walk.py, step 0). `early` is
 *  how many seconds before that moment it was shown; null for every other value.
 *
 *  `invented`: the value was shown while the field should be empty (`goldAt`), which is what the form at its end
 *  counts as invented. */
export function verdict(track, field, time) {
  const walk = track.walks[field.id];
  const index = walk ? walk.displays.findIndex((display) => !display.kept && Math.abs(display.t - time) < SAME_MOMENT) : -1;
  if (index < 0) return { kind: 'unread', display: null, early: null, invented: false };
  const display = walk.displays[index], after = walk.displays[index + 1];
  const early = display.grade < 1 && after && after.kept && after.grade >= 1 ? after.t - display.t : null;
  const change = (track.updates[field.id] || []).find((item) => Math.abs(item[0] - time) < SAME_MOMENT);
  const invented = early === null && Boolean(change) && change[1] !== null && goldAt(field, time).empty;
  return { kind: early !== null ? 'right' : invented ? 'invented' : gradeKind(display), display, early, invented };
}

/** The window on a value the form showed: what its field asks for, the whole value, from when, and how the
 *  evaluation read it. For a value that counts as invented it also holds the form's rules: with the field's prompt,
 *  they are what says whether the value belongs there. */
export function shownWindow(track, field, time) {
  const changes = track.updates[field.id] || [];
  const index = changes.findIndex((change) => Math.abs(change[0] - time) < SAME_MOMENT);
  if (index < 0) return rows([['Field', field.label], asks(field), ['Shows', 'nothing']]);
  const [at, value] = changes[index];
  const next = changes[index + 1];
  const { display, early, invented } = verdict(track, field, at);
  const said = goldAt(field, at).index;
  let graded = track.kind === 'live' ? 'not graded yet' : 'not graded: it was replaced before it was read';
  if (display) {
    const counts = COUNTS[display.kind] || (display.grade > 0 ? 'close to what was said' : 'not what was said');
    graded = `${figure(display.grade, 2)}: ${counts}`;
  }
  if (early !== null) {
    graded = `shown ${figure(early, 1)} s before the words that say it had arrived. For those seconds it is graded `
      + `${figure(display.grade, 2)}, against what had been said until then; from ${clock(at + early, true)} on it counts as `
      + 'right, with no delay.';
  } else if (invented) {
    graded = `${figure(display.grade, 2)}: invented. ${shouldBeEmpty(field, said)}`;
  }
  let shows = value === null ? el('span', { class: 'type' }, 'nothing: the field was cleared') : valueNode(value);
  // a note: sentence by sentence, against what had been said when it was written
  const judged = said >= 0 ? judgedAgainst(track, field, at, said) : null;
  if (judged) {
    shows = el('ul', { class: 'units' }, judged.units.map((unit, k) =>
      el('li', {}, mark(judged.unitOk[k] ? 'right' : 'invented'), el('span', {}, unit))));
  }
  // a value that was said, and that is still on the form after the conversation took it back: what holds when
  // this value leaves the form (or at the end) is a withdrawal that was made after it was shown
  const last = field.steps[goldAt(field, next ? next[0] - SAME_MOMENT : Infinity).index];
  const taken = value !== null && !invented && last && last.value === null && last.said > at ? last : null;
  return rows([
    ['Field', field.label],
    asks(field),
    ['Shows', shows],
    ['From', clock(at, true)],
    ['Until', next ? clock(next[0], true) : track.kind === 'live' ? 'now' : 'the end'],
    ['Grade', graded],
    judged && ['Marks', '✓ the facts said by then support this sentence or item; + they do not, so it counts as invented '
      + '(it may still be true to the conversation: the judge only reads the reference)'],
    taken && ['Then', `the value of this field is withdrawn at ${clock(taken.said, true)}`
      + `${taken.heard ? ` (“… ${taken.heard}”)` : ''}. From then on the field should be empty: what it still shows `
      + 'counts as invented.'],
    (invented || taken) && ['The form’s rules', track.conversation.form.rules || '—'],
  ]);
}

/** A field's state in a word and a line (track.js, `state`). */
export function describe(state, field, track) {
  const { word } = MARKS[state.kind];
  const limit = track.conversation.speed_limit_s;
  const wanted = Array.isArray(state.wanted) && field.type === 'textarea'
    ? `${state.wanted.length} fact${state.wanted.length === 1 ? '' : 's'}` : plain(state.wanted);
  // a note that was judged: its grade is made of two counts, and one word can't say which of them is low
  const judged = field.type === 'textarea' && state.display && state.step >= 0
    ? judgedAgainst(track, field, state.since, state.step) : null;
  if (judged && ['right', 'close', 'wrong'].includes(state.kind)) {
    const said = judged.factOk.filter(Boolean).length, follow = judged.unitOk.filter(Boolean).length;
    return { word, detail: `it says ${said} of ${judged.factOk.length} facts; ${follow} of its ${judged.unitOk.length} `
      + `${field.list ? 'items' : 'sentences'} follow from them (grade ${figure(state.display.grade, 2)})` };
  }
  switch (state.kind) {
    case 'idle':
      return { word, detail: field.steps.length ? `Nothing said yet: first at ${clock(field.steps[0].said)}`
        : 'Nothing is said about it in this conversation' };
    case 'empty':
      return { word, detail: 'as it should be: the value was withdrawn' };
    case 'waiting':
      return { word, detail: `${state.waited.toFixed(1)} s for ${wanted}${state.behind ? ' (it shows the value before)' : ''}`
        + (state.waited >= limit ? ': no speed left' : '') };
    case 'right': {
      if (state.display && state.display.kept) return { word, detail: 'already there when it was said' };
      return { word, detail: state.delay === null || state.delay === undefined ? ''
        : `${Math.max(0, state.delay).toFixed(1)} s after it was said` };
    }
    case 'close':
      return { word, detail: `to ${wanted} (grade ${figure(state.display.grade, 2)})` };
    case 'wrong':
      return { word, detail: `it should show ${wanted}` };
    case 'invented':  // why the field should be empty: the short of `shouldBeEmpty`, which the value's window has in full
      if (state.step >= 0) return { word, detail: `it should be empty since ${clock(field.steps[state.step].said)}: the value was withdrawn` };
      return { word, detail: field.steps.length ? `nothing was said for it yet: first at ${clock(field.steps[0].said)}`
        : 'the labels hold no value for this field' };
    default:
      return { word, detail: track.kind === 'live' ? 'the next scoring reads it' : 'replaced before it was read' };
  }
}
