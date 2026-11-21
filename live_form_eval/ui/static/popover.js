// The window that opens on a value: it holds the whole value, however long, and it can be read.
//
// It opens a moment after the pointer rests on something that has one (or when that gets the keyboard focus), next
// to it. It stays open while the pointer is on the value or inside the window itself, so the pointer can move into
// it and scroll; it closes a moment after the pointer has left both, or with Escape.

const OPEN_AFTER_MS = 140;   // long enough not to flicker while the pointer crosses the form
const CLOSE_AFTER_MS = 320;  // long enough to move the pointer from the value into the window
const GAP = 6;
const EDGE = 12;

export class Popover {
  constructor(node) {
    this.node = node;
    this.builders = new WeakMap();  // element → () => what its window holds (a Node)
    this.owner = null;
    this.opening = null;
    this.closing = null;
    document.addEventListener('pointerover', (event) => this.over(event.target));
    document.addEventListener('focusin', (event) => this.over(event.target, true));
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') this.hide(); });
    window.addEventListener('blur', () => this.hide());
  }

  /** Give `target` a window; `build()` is called each time it opens. */
  on(target, build) {
    target.dataset.pop = '';
    this.builders.set(target, build);
    return target;
  }

  over(element, now = false) {
    if (!(element instanceof Element)) return;
    if (this.node.contains(element)) {  // inside the window: it stays
      clearTimeout(this.closing);
      return;
    }
    const target = element.closest('[data-pop]');
    clearTimeout(this.opening);
    if (!target) {
      if (!this.node.hidden) {
        clearTimeout(this.closing);
        this.closing = setTimeout(() => this.hide(), CLOSE_AFTER_MS);
      }
      return;
    }
    clearTimeout(this.closing);
    if (target === this.owner && !this.node.hidden) return;
    this.opening = setTimeout(() => this.show(target), now ? 0 : OPEN_AFTER_MS);
  }

  show(target) {
    const build = this.builders.get(target);
    if (!build || !target.isConnected) return;
    this.owner = target;
    this.node.replaceChildren(build());
    this.node.hidden = false;
    this.node.scrollTop = 0;
    // next to the value: under it where there is room, else above it; never off the screen
    const at = target.getBoundingClientRect();
    const width = this.node.offsetWidth, height = this.node.offsetHeight;
    const below = at.bottom + GAP + height <= window.innerHeight - EDGE;
    const top = below ? at.bottom + GAP : Math.max(EDGE, at.top - GAP - height);
    const left = Math.min(Math.max(EDGE, at.left), Math.max(EDGE, window.innerWidth - EDGE - width));
    this.node.style.top = `${top}px`;
    this.node.style.left = `${left}px`;
  }

  hide() {
    clearTimeout(this.opening);
    clearTimeout(this.closing);
    this.node.hidden = true;
    this.owner = null;
  }
}
