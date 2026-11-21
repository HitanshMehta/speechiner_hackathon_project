// The form: its rules, then one card per field, filling as the conversation goes.
//
// On top stand the form's rules, the text an extractor gets for the whole form. A card shows the field's prompt
// (what it asks for), what the field holds now, a mark and a line that say how that compares with what was said,
// and two thin lines over the conversation's time: what the field should show (grey) and what it showed (green =
// right, amber = close, red = wrong). The rules and the prompt are what says whether a value belongs in a field:
// they are there to be read when a value is marked wrong or invented. A note or an answer shows its text sentence
// by sentence next to the facts it should state by now. Resting the pointer on a value or a line opens a window
// with all of it; the field's name opens every value of the field over time (the name is a button that covers the
// card, so a click anywhere does).

import { clock, el, figure, lastAt, plain } from './util.js';
import { describe, goldWindow, gradeKind, judgedAgainst, LEGEND, mark, MARKS, shownWindow, stepChange, valueNode, verdict } from './details.js';

const SAME_MOMENT = 1e-6;
const JUST_CHANGED_S = 2;  // a value that changed this recently lights up

/** "Hotel: area" → ["Hotel", "area"]; "Question 1 (QMSum)" → ["QMSum", "Question 1"]; else no group. */
function named(label) {
  const prefixed = label.match(/^(.+?): (.+)$/);
  if (prefixed) return [prefixed[1], prefixed[2]];
  const suffixed = label.match(/^(.+) \((.+)\)$/);
  return suffixed ? [suffixed[2], suffixed[1]] : ['', label];
}

export class FormView {
  constructor(nodes, popover) {
    this.nodes = nodes;  // cards, note, legend, drawer, drawerTitle, drawerBody
    this.popover = popover;
    this.cards = [];
    this.everyField = false;
    this.opened = null;
    this.cameFrom = null;
    this.track = null;
    nodes.legend.replaceChildren(...LEGEND.map((kind) => el('li', {}, mark(kind), `${MARKS[kind].word}: ${MARKS[kind].means}`)));
  }

  load(track) {
    this.track = track;
    this.close();
    const fields = track.fields;
    const names = fields.map((field) => named(field.label));
    const sizes = {};
    for (const [group] of names) sizes[group] = (sizes[group] || 0) + 1;
    this.cards = [];
    this.groups = [];
    const nodes = [];
    let current = null;
    fields.forEach((field, k) => {
      const group = sizes[names[k][0]] > 1 ? names[k][0] : '';  // a group of one is no group
      if (group !== current) {
        current = group;
        const heading = el('h3', { class: 'group' }, group || (this.groups.length ? 'Other fields' : ''));
        heading.hidden = !heading.textContent;
        this.groups.push({ heading, cards: [] });
        nodes.push(heading);
      }
      const card = this.make(field, group ? names[k][1] : field.label);
      this.groups[this.groups.length - 1].cards.push(card);
      this.cards.push(card);
      nodes.push(card.node);
    });
    this.rest = el('p', { class: 'rest' });
    // the form's rules, as an extractor gets them: with each field's prompt, all it is told about what to fill
    const rules = track.conversation.form.rules;
    const told = rules ? el('section', { class: 'rules' }, el('h3', {}, 'The form’s rules'), el('p', {}, rules),
      el('p', { class: 'hint' }, 'What an extractor is told for the whole form (session.form.description). The line under '
        + 'each field’s name is that field’s prompt.')) : null;
    this.nodes.cards.replaceChildren(...[told, ...nodes, this.rest].filter(Boolean));
    this.nodes.cards.scrollTop = 0;
    this.shownCount = -1;
  }

  make(field, name) {
    const text = field.type === 'textarea';
    const card = { field, text, signature: null, lanesOf: null, since: undefined, inPlay: false };
    card.well = el('div', { class: 'well empty' }, '—');
    this.popover.on(card.well, () => (card.since === null || card.since === undefined
      ? el('span', {}, 'The field is empty.') : shownWindow(this.track, field, card.since)));
    card.mark = mark('idle');
    card.word = el('span', { class: 'word' });
    card.detail = el('span', { class: 'detail' });
    card.lanes = el('div', { class: 'lanes' });
    card.opener = el('button', { class: 'label', type: 'button', title: `${field.label}: every value over time` }, name);
    card.opener.addEventListener('click', () => this.open(card));
    const head = el('header', {}, card.opener, el('span', { class: 'type' }, field.type));
    const ask = el('p', { class: 'ask' }, field.prompt);  // what the field asks for
    const state = el('div', { class: 'state' }, card.mark, card.word, card.detail);
    if (text) {
      card.facts = el('ul', { class: 'facts well' });
      card.node = el('article', { class: 'card text' }, head, ask,
        el('div', { class: 'pair' },
          el('div', {}, el('p', { class: 'sub' }, 'On the form'), card.well),
          el('div', {}, el('p', { class: 'sub' }, 'Should say by now'), card.facts)),
        state, card.lanes);
      // every fact of the field, with the first moment it holds
      card.all = [];
      field.steps.forEach((step) => (step.value || []).forEach((fact) => {
        if (!card.all.some((known) => known.text === fact)) card.all.push({ text: fact, said: step.said });
      }));
    } else {
      card.node = el('article', { class: 'card' }, head, ask, card.well, state, card.lanes);
    }
    return card;
  }

  showEvery(every) {
    this.everyField = every;
    this.shownCount = -1;
  }

  /** Draw the form at second t. `moving`: the speed at which the position moves forward by itself (0 = it
   *  doesn't): a value that changes while it moves lights up. */
  render(t, moving) {
    const track = this.track;
    if (!track) return;
    this.nodes.cards.style.setProperty('--now', track.end ? Math.min(1, t / track.end) : 0);
    let inPlay = 0;
    for (const card of this.cards) {
      const state = track.state(card.field, t);
      const waited = state.kind === 'waiting' ? Math.floor(state.waited * 10) : '';
      const signature = [state.kind, state.since, state.step, state.display ? state.display.t : '', waited, track.read].join('|');
      if (signature !== card.signature) {
        card.signature = signature;
        const changed = state.since !== card.since;
        card.since = state.since;
        this.fill(card, state, t);
        if (changed && moving && state.since !== null && t - state.since < JUST_CHANGED_S * Math.max(1, moving)) {
          card.well.classList.remove('changed');
          void card.well.offsetWidth;  // lets the animation start again
          card.well.classList.add('changed');
        }
        const { word, detail } = describe(state, card.field, track);
        const fresh = mark(state.kind);
        card.mark.replaceWith(fresh);
        card.mark = fresh;
        card.word.textContent = word;
        card.detail.textContent = detail;
        card.detail.title = detail;
      }
      this.lanes(card);
      card.inPlay = card.field.steps.length > 0 || (track.updates[card.field.id] || []).length > 0;
      inPlay += card.inPlay;
    }
    if (inPlay !== this.shownCount) {  // which cards are listed
      this.shownCount = inPlay;
      for (const card of this.cards) card.node.hidden = !this.everyField && !card.inPlay;
      for (const group of this.groups) {
        group.heading.hidden = !group.heading.textContent || group.cards.every((card) => card.node.hidden);
      }
      const hidden = this.cards.length - inPlay;
      this.rest.textContent = this.everyField || !hidden ? ''
        : `${hidden} more field${hidden === 1 ? '' : 's'}: nothing is said about ${hidden === 1 ? 'it' : 'them'} in this `
          + `conversation and the form shows nothing there. “All fields” lists ${hidden === 1 ? 'it' : 'them'}.`;
      this.nodes.note.textContent = `${track.conversation.form.name}: ${track.fields.filter((field) => field.steps.length).length} of `
        + `${track.fields.length} fields have something to show in this conversation.`;
    }
    if (this.opened && this.openedRead !== track.read) this.details(this.opened);
  }

  /** What the field holds now. */
  fill(card, state, t) {
    const { field, well } = card;
    well.classList.toggle('empty', state.value === null);
    if (!card.text) {
      well.textContent = state.value === null ? '—' : plain(state.value);
      return;
    }
    const judged = state.value !== null && state.step >= 0 ? judgedAgainst(this.track, field, state.since, state.step) : null;
    if (state.value === null) {
      well.textContent = '—';
    } else if (judged) {  // sentence by sentence: do the facts said so far support it? If not, it counts as invented
      well.replaceChildren(el('ul', { class: 'units' }, judged.units.map((unit, k) =>
        el('li', {}, mark(judged.unitOk[k] ? 'right' : 'invented'), el('span', {}, unit)))));
    } else {
      well.replaceChildren(el('ul', { class: 'units' }, String(state.value).split('\n').filter((line) => line.trim())
        .map((line) => el('li', {}, mark('idle'), el('span', {}, line)))));
    }
    // the facts it should state: those that hold now (found in the text or not), then those still to come
    const holding = Array.isArray(state.wanted) ? state.wanted : [];
    const items = card.all.map((fact) => {
      const at = holding.indexOf(fact.text);
      if (at >= 0) {
        const kind = judged ? (judged.factOk[at] ? 'right' : 'waiting') : state.value === null ? 'waiting' : 'idle';
        return el('li', {}, mark(kind), el('span', {}, fact.text));
      }
      if (fact.said > t) {
        return el('li', { class: 'later' }, mark('idle'), el('span', {}, fact.text, ' ', el('span', { class: 'when' }, `at ${clock(fact.said)}`)));
      }
      return null;  // a fact the conversation changed later: it no longer holds
    });
    card.facts.replaceChildren(...(card.all.length ? items.filter(Boolean)
      : [el('li', { class: 'later' }, mark('idle'), el('span', {}, 'Nothing: the conversation gives no answer, so the field should stay empty.'))]));
  }

  /** The two lines over time: what the field should show, and what it showed. */
  lanes(card) {
    const track = this.track, { field } = card;
    const changes = track.updates[field.id] || [];
    const signature = `${track.read}|${changes.length}|${track.end}|${Boolean(track.status)}`;
    if (signature === card.lanesOf) return;
    card.lanesOf = signature;
    const share = (seconds) => `${(100 * Math.max(0, seconds) / track.end).toFixed(3)}%`;
    const bars = [];
    const bar = (kind, from, to, build) => {
      const node = el('button', { class: `bar ${kind}`, type: 'button', tabindex: '-1', 'aria-hidden': 'true' });
      node.style.left = share(from);
      if (to === null) node.style.right = 'calc((1 - var(--now, 0)) * 100%)';  // still on screen: it grows with the time
      else node.style.width = share(to - from);
      bars.push(this.popover.on(node, build));
    };
    field.steps.forEach((step, index) => {
      if (step.value === null) return;
      const next = field.steps[index + 1];
      bar('gold', step.said, next ? next.said : track.end, () => goldWindow(track, field, index));
    });
    const live = track.kind === 'live' && !track.status;
    const walk = track.walks[field.id];
    let read = -Infinity;  // up to which change the evaluation has graded
    (walk ? walk.displays : []).forEach((display, index) => {
      const at = lastAt(changes, display.t + SAME_MOMENT, (change) => change[0]);
      read = Math.max(read, display.t);
      if (at < 0 || changes[at][1] === null) return;
      const last = index === walk.displays.length - 1 && at === changes.length - 1;
      bar(`shown ${gradeKind(display)}`, display.t, live && last ? null : display.t + display.seconds,
        () => shownWindow(track, field, changes[at][0]));
    });
    changes.forEach(([at, value], index) => {  // what the evaluation has not read (yet)
      if (at <= read + SAME_MOMENT || value === null) return;
      const next = changes[index + 1];
      bar('shown', at, next ? next[0] : live ? null : track.end, () => shownWindow(track, field, at));
    });
    card.lanes.replaceChildren(...bars);
  }

  // ---- every value of one field over time

  open(card) {
    if (!this.opened) this.cameFrom = document.activeElement;
    this.opened = card;
    this.nodes.drawer.hidden = false;
    this.details(card);
    this.nodes.drawer.querySelector('button').focus();
  }

  close() {
    if (this.opened && this.cameFrom && this.cameFrom.isConnected) this.cameFrom.focus();
    this.opened = null;
    this.nodes.drawer.hidden = true;
  }

  details(card) {
    const track = this.track, { field } = card;
    this.openedRead = track.read;
    this.nodes.drawerTitle.textContent = field.label;
    const walk = track.walks[field.id];
    const changes = track.updates[field.id] || [];
    const table = (head, rows) => el('table', { class: 'plain' },
      el('thead', {}, el('tr', {}, head.map((name) => el('th', { scope: 'col' }, name)))), el('tbody', {}, rows));

    const about = el('dl', {},
      el('dt', {}, 'It asks for'), el('dd', {}, field.prompt || '—'),
      el('dt', {}, 'Type'), el('dd', {}, field.type + (field.options.length ? `: ${field.options.join(', ')}`
        : field.range ? `: ${field.range[0]} to ${field.range[1]}` : '')),
      el('dt', {}, 'Publish it as'), el('dd', {}, field.id),
      el('dt', {}, 'The form’s rules'), el('dd', {}, track.conversation.form.rules || '—'));

    const should = field.steps.map((step, index) => {
      const read = walk && walk.steps[index];
      const change = stepChange(field, index);
      const value = !change ? valueNode(step.value, 'withdrawn')
        : el('ul', { class: 'units' }, [...change.added.map((piece) => el('li', {}, '+', el('span', {}, piece))),
          ...change.ended.map((piece) => el('li', {}, '−', el('span', {}, piece)))]);
      const did = !read ? '' : read.shown === null ? 'not shown'
        : `${read.landed ? 'shown' : 'closest'} at ${clock(read.shown, true)}`;
      return el('tr', {}, el('td', { class: 'n' }, clock(step.said, true)), el('td', { class: 'v' }, value),
        el('td', { class: 'v' }, step.heard ? `“… ${step.heard}”` : ''), el('td', { class: 'n' }, did),
        el('td', { class: 'n' }, read && read.delay !== null ? `${figure(Math.max(0, read.delay), 1)} s` : ''),
        el('td', { class: 'n' }, read ? figure(read.score, 2) : ''));
    });

    const showed = changes.map(([at, value], index) => {
      const { display, early, invented } = verdict(track, field, at);
      const next = changes[index + 1];
      const until = next ? next[0] : track.status || track.kind !== 'live' ? track.end : null;
      // a value shown before its words arrived: graded against what had been said until then, right from then on
      const grade = !display ? 'not graded' : early !== null
        ? `${figure(display.grade, 2)} for ${figure(early, 1)} s, then 1.00: it was said`
        : `${figure(display.grade, 2)}${invented ? ': invented' : ''}`;
      return el('tr', {}, el('td', { class: 'n' }, clock(at, true)),
        el('td', { class: 'v' }, valueNode(value, 'cleared')), el('td', { class: 'n' }, grade),
        el('td', { class: 'n' }, until === null ? '' : `${figure(until - at, 1)} s`));
    });

    const pieces = field.type === 'textarea' || field.type === 'multiselect';
    this.nodes.drawerBody.replaceChildren(
      el('section', {}, about),
      el('section', {}, el('h3', {}, 'What it should show'),
        should.length ? table(['Said at', pieces ? 'The step adds (+) or ends (−)' : 'Value', 'Heard', 'The form', 'Delay', 'Score'], should)
          : el('p', {}, 'Nothing: the labels of this conversation hold no value for this field, so it should stay empty.')),
      el('section', {}, el('h3', {}, 'What the form showed'),
        showed.length ? table(['At', 'Value', 'Grade', 'On screen'], showed) : el('p', {}, 'Nothing so far.')));
  }
}
