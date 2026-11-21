// The conversation: what the extractor receives, at the moment it receives it.
//
// On top, the message that arrived last, like a subtitle: the turn so far (a partial) or the finished turn, with
// the words that are new in it standing out. Under it, the conversation as a timeline with three columns: what
// arrived, what should be filled from there on (the gold values those words complete), and what the form filled
// while that was the latest thing it had. Two ways to cut it:
//
//   turns           one line per turn, growing while the turn is spoken
//   every message   one line per partial and finished turn, exactly as the extractor got them. This is the view
//                   that shows why a dictated number is expected piece by piece: it arrives piece by piece.
//
// A partial carries its whole turn so far. A long turn is therefore shown from its last words on, with the words
// that are new; resting the pointer on it opens the whole message.

import { clock, el, lastAt, plain } from './util.js';
import { goldWindow, mark, shownWindow, stepLine, verdict } from './details.js';

const NEAR_THE_END = 48;   // px: closer to the bottom than this, the timeline keeps following the conversation
const JUST_BEFORE = 1e-9;
const SUBTITLE_KEEPS = 170;  // characters of a turn's earlier words that stay in the subtitle
const LINE_KEEPS = 90;       // … and in a line of the timeline

export class TalkView {
  /** `nodes`: caption, said, scroller, rows, follow. `seek(t)` moves the position (a click on a line's time). */
  constructor(nodes, popover, seek) {
    this.nodes = nodes;
    this.popover = popover;
    this.seek = seek;
    this.view = 'turns';
    this.track = null;
    this.pinned = true;
    nodes.scroller.addEventListener('scroll', () => {
      const { scrollHeight, scrollTop, clientHeight } = nodes.scroller;
      this.pinned = scrollHeight - scrollTop - clientHeight < NEAR_THE_END;
      nodes.follow.hidden = this.pinned;
    });
    nodes.follow.addEventListener('click', () => {
      this.pinned = true;
      nodes.follow.hidden = true;
      nodes.scroller.scrollTop = nodes.scroller.scrollHeight;
    });
  }

  load(track) {
    this.track = track;
    this.stream = track.conversation.stream;
    this.roles = track.conversation.speakers;
    this.fields = new Map(track.fields.map((field) => [field.id, field]));
    // every gold step of every field, in the order they are said
    this.gold = track.fields.flatMap((field) => field.steps.map((step, index) => ({ said: step.said, field, index })))
      .sort((a, b) => a.said - b.said);
    this.cut();
  }

  setView(view) {
    this.view = view;
    if (this.track) this.cut();
  }

  /** Cut the stream into the timeline's lines: one per message, or one per turn. */
  cut() {
    this.groups = [];
    this.stream.forEach((message, index) => {
      const open = this.groups[this.groups.length - 1];
      if (this.view === 'turns' && open && this.stream[open.first].turn === message.turn) open.last = index;
      else this.groups.push({ first: index, last: index, start: message.t });
    });
    this.groups.forEach((group, k) => {
      group.from = k === 0 ? 0 : group.start;  // what happened before the first message counts with it
      group.until = k + 1 < this.groups.length ? this.groups[k + 1].start : Infinity;
    });
    this.lines = [];
    this.nodes.rows.replaceChildren();
    this.captioned = undefined;
    this.saidNow = '';
    this.pinned = true;
    this.nodes.follow.hidden = true;
  }

  who(speaker) {
    const unknown = speaker === 'PENDING';
    const title = unknown ? 'too short to tell who speaks' : this.roles[speaker] || 'speaker';
    return el('span', { class: 'who', title }, unknown ? '?' : speaker);
  }

  /** A message in full: its text, speaker by speaker. */
  whole(index) {
    return this.stream[index].parts.flatMap(([speaker, text]) => [speaker ? this.who(speaker) : null, `${text} `]).filter(Boolean);
  }

  /** A message as what it adds: the turn's earlier words (at most the last `keeps` characters of them), then the
   *  words that are new since the turn's message before it. */
  growth(index, keeps) {
    const message = this.stream[index];
    const text = message.parts.map((part) => part[1]).join(' ');
    const before = index > 0 && this.stream[index - 1].turn === message.turn
      ? this.stream[index - 1].parts.map((part) => part[1]).join(' ') : '';
    const kept = text.startsWith(before) ? before : '';
    const cut = kept.length > keeps;
    const earlier = cut ? `… ${kept.slice(kept.length - keeps).replace(/^\S*\s/, '')}` : kept;
    const speaker = message.parts[message.parts.length - 1][0];
    return { cut, nodes: [speaker ? this.who(speaker) : null, el('span', { class: 'old' }, earlier),
                          el('span', { class: 'new' }, text.slice(kept.length))].filter(Boolean) };
  }

  goldChip({ field, index }) {
    const chip = el('button', { class: 'chip', type: 'button', tabindex: '-1' },
      el('b', {}, field.label), el('span', {}, stepLine(field, index)));
    return this.popover.on(chip, () => goldWindow(this.track, field, index));
  }

  filledChip({ t, field: id, value }) {
    const field = this.fields.get(id);
    if (!field) return null;
    const { kind, early } = verdict(this.track, field, t);
    const chip = el('button', { class: 'chip', type: 'button', tabindex: '-1' },
      mark(kind), el('b', {}, field.label), el('span', {}, value === null ? 'cleared' : plain(value)),
      early === null ? null : el('i', {}, `${early.toFixed(1)} s before it was said`));
    return this.popover.on(chip, () => shownWindow(this.track, field, t));
  }

  render(t) {
    const track = this.track;
    if (!track) return;
    const latest = lastAt(this.stream, t, (message) => message.t);

    if (latest !== this.captioned) {  // the subtitle
      this.captioned = latest;
      const caption = this.nodes.caption;
      caption.classList.toggle('quiet', latest < 0);
      if (latest < 0) {
        caption.replaceChildren('Nothing has arrived yet. The first words reach the extractor at '
          + `${this.stream.length ? clock(this.stream[0].t, true) : '–'}.`);
      } else {
        const message = this.stream[latest];
        caption.replaceChildren(
          el('span', { class: 'tag' }, `${clock(message.t, true)} · ${message.kind === 'turn' ? 'the turn is finished' : 'the turn so far'}`),
          ...this.growth(latest, SUBTITLE_KEEPS).nodes);
      }
    }

    // what was said for the form in the last seconds: as long as showing it still earns some speed
    const limit = track.conversation.speed_limit_s;
    const from = lastAt(this.gold, t - limit, (item) => item.said) + 1, to = lastAt(this.gold, t, (item) => item.said) + 1;
    if (`${from}-${to}` !== this.saidNow) {
      this.saidNow = `${from}-${to}`;
      // the newest first: the line holds as many as fit (the timeline below has them all)
      this.nodes.said.replaceChildren(...this.gold.slice(from, to).reverse().map((item) => this.goldChip(item)));
    }

    const count = lastAt(this.groups, t, (group) => group.start) + 1;
    while (this.lines.length > count) this.lines.pop().node.remove();
    while (this.lines.length < count) {
      const group = this.groups[this.lines.length];
      const line = { group, heard: el('td', { class: 'heard' }), want: el('td', { class: 'want' }),
                     got: el('td', { class: 'got' }), message: -1, wanted: '', filled: '' };
      const when = el('button', { class: 'link', type: 'button', title: 'Go to this moment' }, clock(group.start));
      when.addEventListener('click', () => this.seek(group.start));
      line.node = el('tr', {}, el('td', { class: 'when' }, when), line.heard, line.want, line.got);
      this.nodes.rows.append(line.node);
      this.lines.push(line);
      this.paint(line, t, latest);
    }
    // the last lines change with the time; the others only when the evaluation has read more of the form
    const first = track.read === this.read ? Math.max(0, count - 2) : 0;
    this.read = track.read;
    for (let k = first; k < count; k += 1) this.paint(this.lines[k], t, latest);
    if (this.pinned) this.nodes.scroller.scrollTop = this.nodes.scroller.scrollHeight;
  }

  paint(line, t, latest) {
    const { group } = line, track = this.track;
    const message = Math.min(group.last, latest);
    if (message !== line.message) {
      line.message = message;
      const finished = this.stream[message].kind === 'turn';
      if (this.view === 'turns') {
        line.heard.replaceChildren(...this.whole(message), ...(message < group.last ? [el('span', { class: 'type' }, '…')] : []));
      } else if (finished) {
        line.heard.replaceChildren(...this.whole(message), el('span', { class: 'type' }, '(the turn is finished)'));
      } else {
        const { cut, nodes } = this.growth(message, LINE_KEEPS);
        line.heard.replaceChildren(...nodes);
        if (cut) this.popover.on(line.heard, () => el('div', {}, this.whole(message)));
      }
    }
    const upTo = Math.min(t, group.until - JUST_BEFORE);
    const wantFrom = lastAt(this.gold, group.from - JUST_BEFORE, (item) => item.said) + 1;
    const wantTo = lastAt(this.gold, upTo, (item) => item.said) + 1;
    if (`${wantFrom}-${wantTo}` !== line.wanted) {
      line.wanted = `${wantFrom}-${wantTo}`;
      line.want.replaceChildren(...this.gold.slice(wantFrom, wantTo).map((item) => this.goldChip(item)));
    }
    const gotFrom = lastAt(track.all, group.from - JUST_BEFORE, (change) => change.t) + 1;
    const gotTo = lastAt(track.all, upTo, (change) => change.t) + 1;
    const filled = `${gotFrom}-${gotTo}-${gotTo > gotFrom ? track.read : ''}`;
    if (filled !== line.filled) {
      line.filled = filled;
      line.got.replaceChildren(...track.all.slice(gotFrom, gotTo).map((change) => this.filledChip(change)).filter(Boolean));
    }
  }
}
