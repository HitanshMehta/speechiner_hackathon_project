// What is being played: one conversation and what a form showed during it, as far as it is known.
//
// A track is filled at once from a run folder (a review), or event by event (a live run), or stays empty (the
// conversation alone). Everything the page draws is read from it at a position t, in seconds on the extractor's
// clock, the same clock as the evaluation's.

import { isEmpty, lastAt, sameValue } from './util.js';

const SAME_MOMENT = 1e-6;

export class Track {
  /** `kind`: 'live', 'review' or 'alone' (a conversation without a run); `end`: the last second of its time axis. */
  constructor(kind, conversation, end) {
    this.kind = kind;
    this.conversation = conversation;
    this.fields = conversation.form.fields;
    this.end = end;
    this.updates = {};   // field → [[t, value], …]: every visible change of it (null = cleared)
    this.all = [];       // the same changes of all fields in time order: [{t, field, value}]
    this.calls = [];     // every use the run reported, in time order: model calls to fill the form and other uses
    this.spent = [];     // the price after each use
    this.attempts = [];  // how many of the uses so far were model calls to fill the form, after each use
    this.series = [];    // the score so far: [{t, score, completeness, precision, speed, outcome}]
    this.walks = {};     // field → how the evaluation read it (ui/playback.py, walks_payload)
    this.read = 0;       // grows whenever the walks change
    this.final = null;   // the official numbers of a run that ended by itself
    this.status = null;  // how the run ended
    this.planned = 0;    // how many points the score over time of a review will have
    this.scored = kind !== 'review';  // is the score over time complete?
    this.log = [];
  }

  /** A value was published. False when it changes nothing (the same value again). */
  change(t, field, value) {
    const shown = isEmpty(value) ? null : value;
    const list = this.updates[field] || (this.updates[field] = []);
    if (sameValue(list.length ? list[list.length - 1][1] : null, shown)) return false;
    list.push([t, shown]);
    this.all.push({ t, field, value: shown });
    return true;
  }

  /** The changes of a whole run at once (field → [[t, value], …], as the evaluation read them). */
  load(updates) {
    this.updates = updates;
    this.all = Object.entries(updates)
      .flatMap(([field, list]) => list.map(([t, value]) => ({ t, field, value })))
      .sort((a, b) => a.t - b.t);
  }

  /** A use the run reported (ui/playback.py, calls_payload): {t, model, extraction, eur, units: {unit: euros},
   *  cache_read, fresh_input, cache_write, output, seconds, ok}. `extraction`: a model call to fill the form. */
  call(call) {
    this.calls.push(call);
    this.spent.push((this.spent.length ? this.spent[this.spent.length - 1] : 0) + (call.eur || 0));
    this.attempts.push((this.attempts.length ? this.attempts[this.attempts.length - 1] : 0) + (call.extraction ? 1 : 0));
  }

  setWalks(walks) {
    this.walks = walks;
    this.read += 1;
  }

  usesUntil(t) { return lastAt(this.calls, t, (call) => call.t) + 1; }
  /** The model calls made to fill the form by second t. */
  callsUntil(t) { const count = this.usesUntil(t); return count ? this.attempts[count - 1] : 0; }
  /** The other uses by second t. */
  otherUntil(t) { return this.usesUntil(t) - this.callsUntil(t); }
  priceAt(t) { const count = this.usesUntil(t); return count ? this.spent[count - 1] : 0; }
  /** The price by second t, unit by unit: [[unit, euros], …] in the order the units first came up. */
  unitsAt(t) {
    const spent = new Map();
    for (const call of this.calls.slice(0, this.usesUntil(t))) {
      for (const [unit, eur] of Object.entries(call.units || {})) spent.set(unit, (spent.get(unit) || 0) + eur);
    }
    return [...spent];
  }

  /** The point of the score so far at second t: the last one at or before it. */
  scoreAt(t) {
    const index = lastAt(this.series, t, (point) => point.t);
    return index < 0 ? null : this.series[index];
  }

  /** The walk's display that was made at second `time` (a change of the field), if the walk has read it. */
  displayMadeAt(field, time) {
    const walk = this.walks[field];
    return walk ? walk.displays.find((display) => !display.kept && Math.abs(display.t - time) < SAME_MOMENT) : undefined;
  }

  /** One field at second t: what it shows, what it should show, and how the two compare.
   *
   *  kind: 'idle'      nothing is said yet, nothing is shown
   *        'empty'     the value was withdrawn and the field is empty, as it should be
   *        'waiting'   a value was said and the form doesn't show it yet (`behind`: it still shows an earlier one)
   *        'right' | 'close' | 'wrong'   what it shows, as the evaluation grades it
   *        'invented'  it shows something where it should be empty: nothing was said for it (yet), or its value
   *                    was withdrawn. It is what the form at the end counts as invented.
   *        'unread'    it shows something the evaluation has not graded (yet)
   */
  state(field, t) {
    const steps = field.steps;
    const changes = this.updates[field.id] || [];
    const u = lastAt(changes, t, (change) => change[0]);
    const k = lastAt(steps, t, (step) => step.said);
    const value = u < 0 ? null : changes[u][1];
    const wanted = k < 0 ? null : steps[k].value;
    const out = { kind: 'idle', value, since: u < 0 ? null : changes[u][0], wanted, step: k, display: null };
    if (value === null) {
      if (k >= 0 && wanted !== null) {
        out.kind = 'waiting';
        out.waited = t - steps[k].said;
      } else if (k >= 0) {
        out.kind = 'empty';
      }
      return out;
    }
    const walk = this.walks[field.id];
    const d = walk ? lastAt(walk.displays, t, (display) => display.t) : -1;
    const display = d < 0 ? null : walk.displays[d];
    // the walk reads a text at most every few seconds, and a live run's walk is a tick behind
    if (!display || display.t < changes[u][0] - SAME_MOMENT) {
      out.kind = 'unread';
      return out;
    }
    out.display = display;
    if (k < 0 || wanted === null) {  // nothing was said for it so far, or what was said was withdrawn
      out.kind = 'invented';
    } else if (display.grade >= 1) {
      if (display.step !== null && display.step < k) {  // right for an earlier value; a newer one was said since
        out.kind = 'waiting';
        out.behind = true;
        out.waited = t - steps[k].said;
      } else {
        out.kind = 'right';
        const read = display.step !== null ? walk.steps[display.step] : null;
        out.delay = read ? read.delay : null;
      }
    } else {
      out.kind = display.grade > 0 ? 'close' : 'wrong';
    }
    return out;
  }
}
