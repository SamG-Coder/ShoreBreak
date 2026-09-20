// First-person "explore" controls for the full-screen mode: look around with the mouse
// (pointer lock) or touch-drag, walk with WASD / arrows / an on-screen stick, run with Shift.
// Walk -> wade -> surface swim, with buoyancy driven by the rendered water field.

const WALK = 1.45;        // m/s
const RUN = 4.6;          // m/s
const EYE = 1.64;         // eye height above the ground (m)
const CROUCH_EYE = 0.82;
const ACCEL = 8.0;        // velocity response (1/s)
const SWIM = .82, SWIM_FAST = 1.28;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const smooth=(a,b,x)=>{const u=clamp((x-a)/(b-a),0,1);return u*u*(3-2*u);};
const mix=(a,b,t)=>a+(b-a)*t;

export class ExploreControls {
  /**
   * @param {HTMLElement} dom      element that receives pointer events (the canvas)
   * @param {THREE.Camera} camera
   * @param {object} opts          { ground(x, z) -> y, bounds: {xMin,xMax,zMin,zMax}, start: {x, z, yaw, pitch} }
   */
  constructor(dom, camera, opts) {
    this.dom = dom;
    this.camera = camera;
    this.ground = opts.ground;
    this.water = opts.water || (()=>({height:0,slopeX:0,slopeZ:0,flowX:0,flowZ:0}));
    this.bounds = opts.bounds;
    const s = opts.start || {};
    this.pos = { x: s.x ?? 0, z: s.z ?? 4.1 };
    this.yaw = s.yaw ?? 0;            // radians, 0 = looking toward -z (the sea)
    this.pitch = s.pitch ?? -0.19;    // radians, negative = down
    this.vel = { x: 0, z: 0 };
    this.keys = new Set();
    this.locked = false;
    this.bobPhase = 0;
    this.bobAmp = 0;
    this.eyeY = this.ground(this.pos.x, this.pos.z) + EYE;
    this.stick = { x: 0, y: 0, id: null, ox: 0, oy: 0 };
    this.look = { id: null, x: 0, y: 0 };
    this.enabled = true;
    this.motion = opts.reducedMotion ? 0 : 1;
    this.runBlend = 0;
    this.crouched = false;
    this.diving=false;this.diveBlend=0;this.jumpY=0;this.jumpVelocity=0;
    this.depth=Math.max(0,-this.ground(this.pos.x,this.pos.z));
    this.swimBlend=0;this.swimming=false;this.mode='walk';
    this.surfaceY=0;this.surfaceMean=0;this.waterTime=0;
    this.swimPhase=0;this.swimPitch=0;this.swimRoll=0;
    this._bind();
  }

  _bind() {
    const d = this.dom;
    d.addEventListener('click', () => { if (this.enabled && window.__ready && !this.locked && !this.touchUsed) { const request = d.requestPointerLock?.(); request?.catch?.(() => {}); } });
    document.addEventListener('pointerlockchange', () => { this.locked = document.pointerLockElement === d; if (!this.locked) this.keys.clear(); this.onLockChange?.(this.locked); });
    document.addEventListener('mousemove', (e) => {
      if (!this.enabled || !this.locked) return;
      this.yaw -= e.movementX * 0.0022;
      this.pitch -= e.movementY * 0.0022;
      this._clampPitch();
    });
    window.addEventListener('keydown', (e) => {
      if (!this.enabled || !window.__ready || e.target?.closest('input,textarea,select')) return;
      if(e.code==='KeyC'&&!e.ctrlKey&&!e.metaKey&&!e.altKey) { if(!e.repeat)this.toggleCrouch(); e.preventDefault(); return; }
      if(e.target?.closest('button'))return;
      if(e.code==='Space'){if(!e.repeat)this.jump();e.preventDefault();return;}
      this.keys.add(e.code);
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code) && this.locked) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => { this.keys.clear(); this.vel.x = this.vel.z = 0; this.stick.x = this.stick.y = 0; });

    // Touch zones follow the canvas, including the framed presentation.
    d.addEventListener('touchstart', (e) => {
      if (!this.enabled || !window.__ready) return;
      this.touchUsed = true;
      const bounds = d.getBoundingClientRect();
      for (const t of e.changedTouches) {
        if (t.clientX - bounds.left < bounds.width * 0.35 && this.stick.id === null) {
          this.stick.id = t.identifier; this.stick.ox = t.clientX; this.stick.oy = t.clientY; this.stick.x = this.stick.y = 0;
          this.onStick?.(true, t.clientX, t.clientY, 0, 0);
        } else if (this.look.id === null) {
          this.look.id = t.identifier; this.look.x = t.clientX; this.look.y = t.clientY;
        }
      }
      e.preventDefault();
    }, { passive: false });
    d.addEventListener('touchmove', (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === this.stick.id) {
          const dx = t.clientX - this.stick.ox, dy = t.clientY - this.stick.oy;
          const r = Math.min(Math.hypot(dx, dy), 60) / 60, a = Math.atan2(dy, dx);
          this.stick.x = Math.cos(a) * r; this.stick.y = Math.sin(a) * r;
          this.onStick?.(true, this.stick.ox, this.stick.oy, this.stick.x, this.stick.y);
        } else if (t.identifier === this.look.id) {
          this.yaw -= (t.clientX - this.look.x) * 0.005;
          this.pitch -= (t.clientY - this.look.y) * 0.005;
          this._clampPitch();
          this.look.x = t.clientX; this.look.y = t.clientY;
        }
      }
      e.preventDefault();
    }, { passive: false });
    const end = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === this.stick.id) { this.stick.id = null; this.stick.x = this.stick.y = 0; this.onStick?.(false); }
        if (t.identifier === this.look.id) this.look.id = null;
      }
    };
    d.addEventListener('touchend', end);
    d.addEventListener('touchcancel', end);
  }

  _clampPitch() { this.pitch = Math.max(-1.45, Math.min(1.35, this.pitch)); }

  toggleCrouch(){
    if(this.swimming||this.diving){this.diving=!this.diving;this.onDiveChange?.(this.diving);return;}
    this.crouched=!this.crouched;this.onCrouchChange?.(this.crouched);
  }
  jump(){
    if(!this.enabled||this.swimming||this.diving||this.depth>.65||this.jumpY>0||this.jumpVelocity>0)return false;
    if(this.crouched){this.crouched=false;this.onCrouchChange?.(false);}
    this.jumpVelocity=3.25*(1-.4*clamp(this.depth/.65,0,1));return true;
  }
  get effort() { return this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') || Math.hypot(this.stick.x,this.stick.y)>.92; }
  get running() { return !this.crouched&&!this.swimming&&this.depth<.28&&this.ground(this.pos.x,this.pos.z)>-.28&&this.effort; }

  update(dt,waterTime=this.waterTime+dt) {
    dt = Math.min(Math.max(dt || 0, 0), 0.1);
    this.waterTime=Number.isFinite(waterTime)?waterTime:this.waterTime;
    const wave=this.water(this.pos.x,this.pos.z,this.waterTime);
    const g0=this.ground(this.pos.x,this.pos.z);
    const depth=Math.max(0,wave.height-g0);
    this.depth+=(depth-this.depth)*(1-Math.exp(-3*dt));
    if(!this.swimming&&this.depth>1.32)this.swimming=true;
    else if(this.swimming&&this.depth<1.08)this.swimming=false;
    if(this.diving&&depth<.75){this.diving=false;this.onDiveChange?.(false);}
    this.diveBlend+=((this.diving?1:0)-this.diveBlend)*(1-Math.exp(-3.8*dt));
    this.swimBlend+=(smooth(1.02,1.48,this.depth)-this.swimBlend)*(1-Math.exp(-4*dt));
    if(this.swimming&&this.crouched){this.crouched=false;this.onCrouchChange?.(false);}
    const mode=this.diving?'dive':this.swimming?'swim':this.depth>.12?'wade':'walk';
    if(mode!==this.mode){this.mode=mode;this.onModeChange?.(mode);}
    const k = this.keys;
    if (!this.enabled) { k.clear(); this.stick.x = this.stick.y = 0; }
    let fx = 0, fz = 0;
    if (k.has('KeyW') || k.has('ArrowUp')) fz += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) fz -= 1;
    if (k.has('KeyA') || k.has('ArrowLeft')) fx -= 1;
    if (k.has('KeyD') || k.has('ArrowRight')) fx += 1;
    fx += this.stick.x; fz -= this.stick.y;
    const len = Math.hypot(fx, fz);
    if (len > 1) { fx /= len; fz /= len; }
    const wade=clamp((this.depth-.02)/.93,0,1);
    const landSpeed=(this.crouched?.72:this.running?RUN:WALK)*(1-.65*wade);
    const speed=mix(landSpeed,this.effort?SWIM_FAST:SWIM,this.swimBlend);
    // desired velocity in world xz (yaw 0 looks toward -z)
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    this.surfaceMean+=(wave.height-this.surfaceMean)*(1-Math.exp(-.6*dt));
    const orbital=clamp((wave.height-this.surfaceMean)*Math.sqrt(9.81/Math.max(this.depth,.5))*.38,-.38,.38);
    const tx = (fx * cy - fz * sy) * speed+clamp(wave.flowX||0,-.35,.35)*this.swimBlend;
    let tz = (-fx * sy - fz * cy) * speed+clamp((wave.flowZ||0)+orbital,-.42,.42)*this.swimBlend;
    // Ease outward movement over the last metre instead of striking a hard edge.
    if(tz<0)tz*=smooth(0,1.2,this.pos.z-this.bounds.zMin);
    const response = mix(len > 0.01 ? ACCEL : 13, len>.01?2.4:1.8,this.swimBlend);
    const a = 1 - Math.exp(-response * dt);
    // Integrate the exponential response exactly: distance does not depend on FPS.
    const moveX = tx * dt + (this.vel.x - tx) * a / response;
    const moveZ = tz * dt + (this.vel.z - tz) * a / response;
    this.vel.x += (tx - this.vel.x) * a;
    this.vel.z += (tz - this.vel.z) * a;
    // walking into the swash is slower (water drag on the legs)
    let nx = this.pos.x + moveX;
    let nz = this.pos.z + moveZ;
    const b = this.bounds;
    nx = Math.max(b.xMin, Math.min(b.xMax, nx));
    nz = Math.max(b.zMin, Math.min(b.zMax, nz));
    const distance = Math.hypot(nx - this.pos.x, nz - this.pos.z);
    this.pos.x = nx; this.pos.z = nz;
    // A stabilized human gaze: one smooth vertical cycle per footfall, no
    // absolute-sine impacts or abrupt amplitude jump when Shift is pressed.
    const sp = dt > 0 ? distance / dt : 0;
    this.runBlend += (Math.max(0, Math.min(1, (sp - WALK) / (RUN - WALK))) - this.runBlend) * (1 - Math.exp(-5 * dt));
    const stride = 0.80 + 0.92 * this.runBlend;
    this.bobPhase = (this.bobPhase + 2 * Math.PI * distance / stride) % (Math.PI * 4);
    const targetAmp = Math.min(sp / WALK, 1) * (0.006 + this.runBlend * 0.005) * this.motion * (1 - .7 * wade)*(1-this.swimBlend);
    this.bobAmp += (targetAmp - this.bobAmp) * (1 - Math.exp(-8 * dt));
    const bobY = -Math.cos(this.bobPhase) * this.bobAmp;
    const bobX = Math.sin(this.bobPhase * 0.5) * this.bobAmp * 0.42;
    const nextWave=this.water(this.pos.x,this.pos.z,this.waterTime);
    // The waterline drives vertical movement even at rest. Footfalls fade out
    // completely; only a tiny stroke sway is added while actively swimming.
    this.surfaceY+=(nextWave.height-this.surfaceY)*(1-Math.exp(-(this.motion?7:16)*dt));
    const ground=this.ground(this.pos.x,this.pos.z);
    const gy=Math.max(this.surfaceY+.32,ground+(this.crouched?CROUCH_EYE:EYE));
    const eyeTarget=Math.max(ground+.24,mix(gy,this.surfaceY+.34,Math.max(this.swimBlend,this.diveBlend))-1.02*this.diveBlend);
    this.eyeY+=(eyeTarget-this.eyeY)*(1-Math.exp(-mix(9,12,this.swimBlend)*dt));
    // Ballistic foot jump. Integrate gravity exactly, then settle at ground;
    // buoyancy handles the landing if the player jumps into deeper water.
    if(this.jumpY>0||this.jumpVelocity>0){
      this.jumpY+=this.jumpVelocity*dt-.5*9.81*dt*dt;this.jumpVelocity-=9.81*dt;
      if(this.jumpY<=0||this.swimming){this.jumpY=0;this.jumpVelocity=0;}
    }
    this.swimPhase=(this.swimPhase+dt*2*Math.PI*(this.effort?.68:.48)*Math.min(len,1))%(2*Math.PI);
    const slopeForward=-(nextWave.slopeX||0)*sy-(nextWave.slopeZ||0)*cy;
    const slopeSide=(nextWave.slopeX||0)*cy-(nextWave.slopeZ||0)*sy;
    const tilt=this.motion*this.swimBlend;
    const pitchTarget=clamp(slopeForward*.10,-.035,.035)*tilt;
    const rollTarget=(clamp(-slopeSide*.07,-.018,.018)+.006*Math.sin(this.swimPhase)*Math.min(len,1))*tilt;
    this.swimPitch+=(pitchTarget-this.swimPitch)*(1-Math.exp(-4*dt));
    this.swimRoll+=(rollTarget-this.swimRoll)*(1-Math.exp(-4*dt));

    const cam = this.camera;
    cam.rotation.order = 'YXZ';
    cam.position.set(this.pos.x + bobX * cy, this.eyeY + bobY+this.jumpY, this.pos.z - bobX * sy);
    cam.rotation.set(this.pitch+this.swimPitch, this.yaw, -bobX * 0.02+this.swimRoll);
    cam.updateMatrixWorld();
    return sp;
  }
}
