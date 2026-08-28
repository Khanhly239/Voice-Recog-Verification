/* Live mouth-landmark overlay for the KYC camera.
 *
 * Runs MediaPipe FaceLandmarker in the browser (a server round-trip cannot keep
 * up with a 25-30 fps overlay) and draws the lip contour on a canvas above the
 * video, plus a syllable counter.
 *
 * What this does and does NOT do, deliberately:
 *   - It shows the user that their mouth is being tracked, and counts how many
 *     times they open their mouth, which should match the number of digits in
 *     the challenge (every Vietnamese digit word is one syllable).
 *   - It does NOT decide which digit was spoken. Our visual digit classifier
 *     currently reaches only ~22% macro accuracy, far too low to judge
 *     correctness; the authoritative content check stays with the ASR, which is
 *     already a hard gate in the verification pipeline.
 */

const MP_VERSION = "0.10.14";
const MODEL_URL = "/api/v1/assets/face_landmarker.task";

// MediaPipe FaceMesh lip ring indices (same set the Python pipeline uses).
const OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   409, 270, 269, 267, 0, 37, 39, 40, 185];
const INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                   415, 310, 311, 312, 13, 82, 81, 80, 191];
const UPPER_LIP_C = 13, LOWER_LIP_C = 14, EYE_L = 33, EYE_R = 263;

// Openness thresholds (normalized by interocular distance) with hysteresis, so
// one syllable is counted once instead of flickering around a single threshold.
const OPEN_ON = 0.22, OPEN_OFF = 0.14;

export class MouthOverlay {
  constructor(video, canvas, opts = {}) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onUpdate = opts.onUpdate || (() => {});
    this.landmarker = null;
    this.running = false;
    this.lastTs = -1;
    this.isOpen = false;
    this.syllables = 0;
    this.faceSeen = false;
  }

  async init() {
    const vision = await import(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`);
    const fileset = await vision.FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`);
    this.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  reset() { this.syllables = 0; this.isOpen = false; }

  start() {
    if (!this.landmarker || this.running) return;
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      this._frame();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() { this.running = false; }

  _frame() {
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return;
    if (this.canvas.width !== v.videoWidth) {
      this.canvas.width = v.videoWidth;
      this.canvas.height = v.videoHeight;
    }
    // MediaPipe requires strictly increasing timestamps.
    const ts = Math.max(performance.now(), this.lastTs + 1);
    this.lastTs = ts;

    let res;
    try { res = this.landmarker.detectForVideo(v, ts); } catch (_) { return; }

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const lm = res && res.faceLandmarks && res.faceLandmarks[0];
    this.faceSeen = !!lm;
    if (!lm) {
      this.onUpdate({ face: false, openness: 0, syllables: this.syllables });
      return;
    }

    const W = this.canvas.width, H = this.canvas.height;
    const px = (i) => [lm[i].x * W, lm[i].y * H];

    const [ex1, ey1] = px(EYE_L), [ex2, ey2] = px(EYE_R);
    const interocular = Math.hypot(ex2 - ex1, ey2 - ey1) || 1;
    const [ux, uy] = px(UPPER_LIP_C), [lx, ly] = px(LOWER_LIP_C);
    const openness = Math.hypot(lx - ux, ly - uy) / interocular;

    // Hysteresis: count a syllable on the closed -> open transition only.
    if (!this.isOpen && openness >= OPEN_ON) {
      this.isOpen = true;
      this.syllables += 1;
    } else if (this.isOpen && openness <= OPEN_OFF) {
      this.isOpen = false;
    }

    this._drawRing(OUTER_LIP, px, this.isOpen ? "#4ade80" : "#38bdf8", 2.5);
    this._drawRing(INNER_LIP, px, "#f472b6", 1.5);
    for (const i of OUTER_LIP) {
      const [x, y] = px(i);
      ctx.fillStyle = this.isOpen ? "#4ade80" : "#38bdf8";
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
    }

    this.onUpdate({
      face: true,
      openness,
      open: this.isOpen,
      syllables: this.syllables,
      faceRatio: interocular / W,
    });
  }

  _drawRing(idx, px, color, width) {
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    idx.forEach((i, k) => {
      const [x, y] = px(i);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
}
