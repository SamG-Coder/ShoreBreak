// Retain fractional time instead of rounding every display frame up to a solver step.
export class FixedClock {
  constructor(dt, maxSteps = 12) { this.dt = dt; this.maxSteps = maxSteps; this.remainder = 0; }
  reset() { this.remainder = 0; }
  advance(elapsed, speed, step) {
    this.remainder += Math.min(Math.max(elapsed, 0), 0.1) * speed;
    const n = Math.min(this.maxSteps, Math.floor((this.remainder + 1e-10) / this.dt));
    for (let i = 0; i < n; i++) step();
    this.remainder -= n * this.dt;
    this.remainder = Math.min(Math.max(this.remainder, 0), this.dt * this.maxSteps);
    return n;
  }
}
