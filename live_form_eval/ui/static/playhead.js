// The position in the conversation, in seconds, and what moves it.
//
// - Reviewing a run: the page's own clock at the chosen speed. With the sound on, the recording leads instead: the
//   position IS the recording's position, so what is shown and what is heard can't drift apart.
// - A live run: the extractor's process owns the clock. The page follows it between its ticks, and the recording
//   is kept on it.
//
// The recording's second 0 is the conversation's second 0 (data/common/audio.py).

const LIVE_DRIFT_S = 0.15;  // in a live run, the recording is moved when it is further than this from the clock

export class Playhead {
  constructor(audio) {
    this.audio = audio;
    this.end = 0;         // the last second that can be reached
    this.rate = 1;
    this.playing = false;
    this.live = false;
    this.sound = false;   // the person wants the recording
    this.soundUpTo = 4;   // … which is only played up to this speed
    this.base = 0;        // the position at the moment `stamp`
    this.stamp = 0;
  }

  /** Is the recording there and playable at this speed? */
  hears() {
    return this.sound && this.rate <= this.soundUpTo && Boolean(this.audio.getAttribute('src'));
  }

  /** Does the recording lead the clock right now? (Reviewing, sound on, and the recording is really running.) */
  led() {
    const audio = this.audio;
    return !this.live && this.hears() && !audio.paused && !audio.ended && !audio.seeking && audio.readyState >= 2;
  }

  now() {
    if (!this.playing) return this.base;
    if (this.led()) {
      this.base = this.audio.currentTime;
      this.stamp = performance.now();
      return this.base;
    }
    const t = this.base + (performance.now() - this.stamp) / 1000 * this.rate;
    if (t >= this.end && !this.live) {  // a review stops at its end; a live run ends when its process says so
      this.playing = false;
      this.base = this.end;
      this.audio.pause();
      return this.end;
    }
    return Math.min(t, this.end);
  }

  /** Bring the recording to the clock: its speed, its position, playing or not. */
  tune() {
    const audio = this.audio;
    const within = !Number.isFinite(audio.duration) || this.base < audio.duration;
    if (!this.hears() || !this.playing || !within) {
      audio.pause();
      return;
    }
    audio.playbackRate = this.rate;
    if (Math.abs(audio.currentTime - this.base) > 0.05) audio.currentTime = this.base;
    audio.play().catch(() => {});  // refused until the person has clicked something: the clock runs on without sound
  }

  hold(t) {
    this.base = t;
    this.stamp = performance.now();
  }

  play() {
    if (this.base >= this.end) this.base = 0;
    this.stamp = performance.now();
    this.playing = true;
    this.tune();
  }

  pause() {
    this.hold(this.now());
    this.playing = false;
    this.audio.pause();
  }

  seek(t) {
    this.hold(Math.min(this.end, Math.max(0, t)));
    this.tune();
  }

  setRate(rate) {
    this.hold(this.now());
    this.rate = rate;
    this.tune();
  }

  setSound(on) {
    this.hold(this.now());
    this.sound = on;
    this.tune();
  }

  /** A live run's clock has started. */
  startLive(rate) {
    this.live = true;
    this.rate = rate;
    this.hold(0);
    this.playing = true;
    this.tune();
  }

  /** A tick of a live run's clock: follow it from here, and keep the recording on it. */
  follow(t) {
    this.hold(t);
    if (this.hears() && Math.abs(this.audio.currentTime - t) > LIVE_DRIFT_S * Math.max(1, this.rate)) {
      this.audio.currentTime = t;
    }
  }

  /** A live run is over at second `t`: from now on the position only moves when the person moves it. */
  endLive(t) {
    this.live = false;
    this.playing = false;
    this.base = Math.min(this.end, t);
    this.audio.pause();
  }

  /** Back to the start of another conversation, of length `end`. */
  reset(end) {
    this.live = false;
    this.playing = false;
    this.end = end;
    this.base = 0;
    this.audio.pause();
  }
}
