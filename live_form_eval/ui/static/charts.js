// One small graph over the conversation's time: a measure so far, with a mark at the current position.
//
// The marks follow the report's rules (evaluation/report/style.py): one 2 px line in the series colour, hairline
// grid, an end dot with a ring in the surface colour, ink-coloured text. What lies after the current position (a
// reviewed run is known to its end) is drawn faint, so the line doubles as a map of where to jump to. The value
// under the pointer is read out for every graph at once (app.js), and every point is in the table as well.

import { clock, lastAt } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
let made = 0;

function shape(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.append(node);
  return node;
}

export class Trend {
  /** `max`: the top of the scale (null = the largest value); `steps`: a value holds until the next point (a
   *  price), instead of a line from point to point; `top(max)`: how the top of the scale is written. */
  constructor(svg, { max = null, steps = false, top = String } = {}) {
    this.svg = svg;
    this.max = max;
    this.steps = steps;
    this.top = top;
    this.id = `trend${made += 1}`;
    this.points = [];   // [{t, v}] in time order; v null = nothing to measure yet
    this.ticks = [];    // times marked on the baseline (model calls)
    this.end = 1;
    this.until = 0;     // up to when the last value is known to hold
    this.at = 0;
    this.hover = null;
    new ResizeObserver(() => this.draw()).observe(svg);
  }

  set(points, until, ticks = []) {
    this.points = points;
    this.until = until;
    this.ticks = ticks;
    this.draw();
  }

  /** The value at second t: the last point at or before it (undefined before the first). */
  valueAt(t) {
    const index = lastAt(this.points, t, (point) => point.t);
    return index < 0 ? undefined : this.points[index].v;
  }

  /** The second under a pointer at clientX. */
  timeAt(clientX) {
    const box = this.svg.getBoundingClientRect();
    return Math.min(this.end, Math.max(0, (clientX - box.left - this.left) / (this.right - this.left) * this.end));
  }

  draw() {
    const width = this.svg.clientWidth, height = this.svg.clientHeight;
    if (!width || !height) return;
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.svg.replaceChildren();
    this.left = 3; this.right = width - 3;
    const top = 13, bottom = height - 13;  // room for the scale's top above the plot and for the times under it
    const peak = this.max ?? Math.max(1e-9, ...this.points.map((point) => point.v || 0));
    this.x = (t) => this.left + (this.right - this.left) * Math.min(1, Math.max(0, t / this.end));
    this.y = (v) => bottom - (bottom - top) * Math.min(1, v / peak);

    const group = shape('g', { class: 'trend' }, this.svg);
    for (const level of [1, 0.5]) {
      shape('line', { class: 'grid', x1: this.left, x2: this.right, y1: this.y(level * peak), y2: this.y(level * peak) }, group);
    }
    shape('line', { class: 'base', x1: this.left, x2: this.right, y1: bottom, y2: bottom }, group);
    for (const t of this.ticks) {
      shape('line', { class: 'rug', x1: this.x(t), x2: this.x(t), y1: bottom, y2: bottom + 4 }, group);
    }

    let path = '', drawing = false, last = null;
    for (const point of this.points) {
      if (point.v === null || point.v === undefined) { drawing = false; continue; }
      const x = this.x(point.t).toFixed(1), y = this.y(point.v).toFixed(1);
      path += !drawing ? `M${x} ${y}` : this.steps ? `H${x}V${y}` : `L${x} ${y}`;
      drawing = true;
      last = point;
    }
    if (drawing && last && this.until > last.t) path += `H${this.x(this.until).toFixed(1)}`;

    // the same line twice: up to the current position, and faint after it
    this.past = shape('rect', { x: 0, y: 0, width: 0, height }, shape('clipPath', { id: `${this.id}a` }, group));
    this.ahead = shape('rect', { x: 0, y: 0, width, height }, shape('clipPath', { id: `${this.id}b` }, group));
    shape('path', { class: 'line', d: path, 'clip-path': `url(#${this.id}a)` }, group);
    shape('path', { class: 'line ahead', d: path, 'clip-path': `url(#${this.id}b)` }, group);

    this.hoverLine = shape('line', { class: 'hover', y1: top - 4, y2: bottom, visibility: 'hidden' }, group);
    this.nowLine = shape('line', { class: 'now', y1: top - 4, y2: bottom }, group);
    this.hoverDot = shape('circle', { class: 'dot', r: 3.5, visibility: 'hidden' }, group);
    this.nowDot = shape('circle', { class: 'dot', r: 4 }, group);

    shape('text', { x: this.left, y: 8 }, group).textContent = this.top(peak);
    shape('text', { x: this.left, y: height - 2 }, group).textContent = clock(0);
    shape('text', { x: this.right, y: height - 2, 'text-anchor': 'end' }, group).textContent = clock(this.end);
    this.place();
  }

  place() {
    if (!this.nowLine) return;
    const put = (line, dot, t) => {
      const shown = t !== null && t !== undefined;
      line.setAttribute('visibility', shown ? 'visible' : 'hidden');
      const value = shown ? this.valueAt(t) : undefined;
      dot.setAttribute('visibility', shown && value !== null && value !== undefined ? 'visible' : 'hidden');
      if (!shown) return;
      const x = this.x(t);
      line.setAttribute('x1', x);
      line.setAttribute('x2', x);
      if (value !== null && value !== undefined) {
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', this.y(value));
      }
    };
    put(this.nowLine, this.nowDot, this.at);
    put(this.hoverLine, this.hoverDot, this.hover);
    const x = this.x(this.at);
    this.past.setAttribute('width', x);
    this.ahead.setAttribute('x', x);
  }

  setNow(t) { this.at = t; this.place(); }
  setHover(t) { this.hover = t; this.place(); }
  setEnd(end) { this.end = Math.max(1e-6, end); this.draw(); }
}
