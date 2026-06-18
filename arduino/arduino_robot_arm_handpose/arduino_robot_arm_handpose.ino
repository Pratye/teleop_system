/**
 * Hand-pose driven robot arm (no physical glove).
 *
 * Hardware:
 * - Arduino Uno
 * - PCA9685 on I2C (SDA=A4, SCL=A5)
 * - 6x MG996R continuous-rotation servos on channels 0..5
 * - A4988 + NEMA17 base (DIR=D4, STEP=D5, EN=D6)
 *
 * Control model: ABSOLUTE positioning.
 * Each incoming H field is a normalized target in [-1..1]; the sketch maps it to
 * an absolute servo position around neutralDeg[i] using SPAN_*. Motion is rate-
 * limited by MAX_DELTA_PER_FRAME for smoothness. Stepper base tracks an absolute
 * step count from home and seeks toward base*BASE_MAX_STEPS_FROM_HOME each frame.
 *
 * Serial protocol @115200, newline-terminated:
 *
 *   H hx hy hz roll pinch base
 *
 * where:
 *   hx,hy,hz  : hand position normalized to [-1..1]
 *   roll      : wrist roll normalized to [-1..1]
 *   pinch     : grip openness [0..1]
 *   base      : base command normalized to [-1..1]
 *
 * Example:
 *   H 0.10 -0.35 0.20 0.05 0.70 -0.40
 *
 * Optional tuning commands:
 *   N n0 n1 n2 n3 n4 n5     (set neutral stop points for channels 0..5)
 *   S id pct                 (directly command one channel speed, pct [-100..100])
 *   B onoff                  (enable/disable base motor command from pose stream)
 */

#include <Wire.h>
#include "HCPCA9685.h"
#include <stdlib.h>
#include <string.h>

const unsigned long BAUD = 115200;
#define I2CAdd 0x40

// HCPCA9685 native servo position scale is 0..450 (covers full pulse-width range).
// We reason in real degrees (0..180) everywhere in this sketch and convert to the
// 0..450 library units at the single write boundary (posToHC). The HC clamp below
// guards against pulse widths that would slam the servo into a hard mechanical stop.
const int HC_POS_MIN = 10;
const int HC_POS_MAX = 440;
const float DEG_TO_HC = 450.0f / 180.0f;

const uint8_t SERVO_COUNT = 6;
const uint8_t SERVO_CHANNELS[SERVO_COUNT] = {0, 1, 2, 3, 4, 5};

// A4988 (pinout unchanged; step profile aligned to reference sketch)
const uint8_t PIN_STEPPER_STEP = 5;
const uint8_t PIN_STEPPER_DIR = 4;
const uint8_t PIN_STEPPER_EN = 6;  // LOW=enabled
const int STEP_PULSE_US = 4500;
const int MAX_STEPS_PER_FRAME = 120;
bool baseEnabled = true;            // enabled for Left/Right gesture control
const float BASE_DEADZONE = 0.15f;  // must be < 0.244 (Left/Right gesture value)

// Parking (home) positions in HCPCA9685 units. Sketch self-aligns here on boot
// and these are the reset targets after a watchdog timeout.
float neutralDeg[SERVO_COUNT] = {90, 90, 90, 90, 90, 90};

// Per-joint min/max in real degrees (0..180 servo travel).
const float POS_MIN[SERVO_COUNT] = {0, 0, 0, 0, 0, 0};
const float POS_MAX[SERVO_COUNT] = {180, 180, 180, 180, 180, 180};

// Per-joint travel from neutral at full-scale input (|value|=1.0) in DEGREES.
// target[i] = clamp(neutralDeg[i] + value * SPAN[i], POS_MIN[i], POS_MAX[i])
// Tune these to match the physical range of motion you want each joint to reach.
const float SPAN_SH = 40.0f;    // shoulder pair (ch0/1) — mirrored
const float SPAN_EL = 80.0f;    // elbow (ch2)
const float SPAN_WY = 80.0f;    // wrist Y (ch3)
const float SPAN_WX = 80.0f;    // wrist X (ch4)
const float SPAN_GR = 24.0f;    // gripper (ch5) — half-range; pinch 0..1 spans -SPAN..+SPAN

// Maximum change in commanded position per incoming pose frame (degrees).
// Keeps motion smooth even if the IK target jumps suddenly.
const float MAX_DELTA_PER_FRAME = 10.0f;

const unsigned long WATCHDOG_STOP_MS = 5000;  // longer timeout — no need to slam to neutral quickly

// Hand -> arm mapping gains (normalized input to normalized speed).
const float G_SHOULDER = 1.0f;  // from hy
const float G_ELBOW = 1.0f;     // from hz
const float G_WRIST_Y = 1.0f;   // from hx
const float G_WRIST_X = 1.0f;   // from roll
const float G_GRIP = 1.0f;      // from pinch
const float G_BASE = 1.0f;      // from base

HCPCA9685 HCPCA9685(I2CAdd);
float outDeg[SERVO_COUNT] = {90, 90, 90, 90, 90, 90};
unsigned long lastRxMs = 0;

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

float deadzone(float x, float dz) {
  if (x > -dz && x < dz) return 0.0f;
  return x;
}

// Convert real degrees (0..180) to HCPCA9685 native units (0..450) at the write
// boundary. All other state in this sketch is in degrees.
int posToHC(float deg) {
  float hc = deg * DEG_TO_HC;
  if (hc < (float)HC_POS_MIN) hc = (float)HC_POS_MIN;
  if (hc > (float)HC_POS_MAX) hc = (float)HC_POS_MAX;
  return (int)(hc + 0.5f);
}

void writeAllServos() {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    HCPCA9685.Servo(SERVO_CHANNELS[i], posToHC(outDeg[i]));
  }
}

// Watchdog "stop": just leave all servos where they currently are.
// (Position servos hold position passively; no need to re-park to neutral on every drop.)
void stopAllServos() {
  writeAllServos();
}

// Move outDeg[id] toward `target` by at most MAX_DELTA_PER_FRAME this frame.
// Keeps servo motion smooth and bounded even when the IK target jumps.
void slewToward(uint8_t id, float target) {
  target = clampf(target, POS_MIN[id], POS_MAX[id]);
  float diff = target - outDeg[id];
  if (diff > MAX_DELTA_PER_FRAME) diff = MAX_DELTA_PER_FRAME;
  else if (diff < -MAX_DELTA_PER_FRAME) diff = -MAX_DELTA_PER_FRAME;
  outDeg[id] += diff;
}

// Smoothly ramp every servo from a starting angle to its parking (neutralDeg) position.
// Equivalent to the reference sketch's wakeUp(): each servo sweeps in small steps so
// the arm self-aligns to a known home pose on startup instead of jumping there abruptly.
void wakeUp() {
  const uint8_t startAngles[SERVO_COUNT] = {90, 90, 90, 90, 90, 90};
  for (uint8_t i = 0; i < SERVO_COUNT; i++) outDeg[i] = (float)startAngles[i];
  writeAllServos();
  delay(300);

  // Sweep each channel toward its neutral one at a time so power draw stays low.
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    float from = (float)startAngles[i];
    float to = neutralDeg[i];
    int steps = 30;
    for (int s = 1; s <= steps; s++) {
      outDeg[i] = from + (to - from) * (float)s / (float)steps;
      HCPCA9685.Servo(SERVO_CHANNELS[i], posToHC(outDeg[i]));
      delay(20);
    }
    outDeg[i] = to;
    HCPCA9685.Servo(SERVO_CHANNELS[i], posToHC(outDeg[i]));
    delay(120);
  }
}

// Stepper base in absolute mode: track current step count from home,
// compute target step count from base ∈ [-1..1], step toward target with
// a per-frame cap so big input jumps don't stall the loop.
const long BASE_MAX_STEPS_FROM_HOME = 800;  // travel range each side of home
long baseCurrentSteps = 0;

void stepBase(float baseNorm) {
  if (!baseEnabled) return;
  baseNorm = clampf(baseNorm, -1.0f, 1.0f);
  long target = (long)(baseNorm * (float)BASE_MAX_STEPS_FROM_HOME);
  long delta = target - baseCurrentSteps;
  if (delta == 0) return;
  long n = (delta > 0) ? delta : -delta;
  if (n > MAX_STEPS_PER_FRAME) n = MAX_STEPS_PER_FRAME;
  digitalWrite(PIN_STEPPER_DIR, (delta > 0) ? HIGH : LOW);
  for (long i = 0; i < n; i++) {
    digitalWrite(PIN_STEPPER_STEP, HIGH);
    delayMicroseconds(STEP_PULSE_US);
    digitalWrite(PIN_STEPPER_STEP, LOW);
    delayMicroseconds(STEP_PULSE_US);
  }
  baseCurrentSteps += (delta > 0) ? n : -n;
}

bool parse7(char *line, float out[7]) {
  char *tok = strtok(line, " \t,");
  for (int i = 0; i < 7; i++) {
    if (!tok) return false;
    if (i == 0) {
      if (tok[0] != 'H' && tok[0] != 'h') return false;
      out[0] = 0.0f;
    } else {
      out[i] = (float)atof(tok);
    }
    tok = strtok(NULL, " \t,");
  }
  return true;
}

bool parseNeutral(char *line, float out[6]) {
  char *tok = strtok(line, " \t,");
  if (!tok || (tok[0] != 'N' && tok[0] != 'n')) return false;
  for (int i = 0; i < 6; i++) {
    tok = strtok(NULL, " \t,");
    if (!tok) return false;
    out[i] = (float)atof(tok);
  }
  return true;
}

bool parseDirectSpeed(char *line, int *id, float *pct) {
  char *tok = strtok(line, " \t,");
  if (!tok || (tok[0] != 'S' && tok[0] != 's')) return false;
  tok = strtok(NULL, " \t,");
  if (!tok) return false;
  *id = atoi(tok);
  tok = strtok(NULL, " \t,");
  if (!tok) return false;
  *pct = (float)atof(tok);
  return true;
}

bool parseBaseEnable(char *line, int *onoff) {
  char *tok = strtok(line, " \t,");
  if (!tok || (tok[0] != 'B' && tok[0] != 'b')) return false;
  tok = strtok(NULL, " \t,");
  if (!tok) return false;
  *onoff = atoi(tok);
  return true;
}

void applyHandPose(float hx, float hy, float hz, float roll, float pinch, float base) {
  // Absolute-target mode: each H field specifies WHERE the joint should be, not how
  // it should drift. target = neutral + value*span, then slew toward target with a
  // per-frame cap so motion stays smooth.
  float sh = clampf(-hy * G_SHOULDER, -1.0f, 1.0f);
  float el = clampf(hz * G_ELBOW, -1.0f, 1.0f);
  float wy = clampf(hx * G_WRIST_Y, -1.0f, 1.0f);
  float wx = clampf(roll * G_WRIST_X, -1.0f, 1.0f);
  float gr = clampf(pinch, 0.0f, 1.0f);            // 0=closed, 1=open
  float bz = clampf(base * G_BASE, -1.0f, 1.0f);

  // Shoulder pair: ch0 and ch1 mirror each other.
  slewToward(0, neutralDeg[0] + sh * SPAN_SH);
  slewToward(1, neutralDeg[1] - sh * SPAN_SH);
  slewToward(2, neutralDeg[2] + el * SPAN_EL);
  slewToward(3, neutralDeg[3] + wy * SPAN_WY);
  slewToward(4, neutralDeg[4] + wx * SPAN_WX);
  // Gripper: pinch 0..1 → neutral - SPAN..neutral + SPAN
  slewToward(5, neutralDeg[5] + (gr * 2.0f - 1.0f) * SPAN_GR);

  writeAllServos();
  stepBase(bz);
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin();
  HCPCA9685.Init(SERVO_MODE);
  HCPCA9685.Sleep(false);
  delay(10);

  pinMode(PIN_STEPPER_STEP, OUTPUT);
  pinMode(PIN_STEPPER_DIR, OUTPUT);
  pinMode(PIN_STEPPER_EN, OUTPUT);
  digitalWrite(PIN_STEPPER_EN, LOW);

  wakeUp();
  Serial.println(F("Hand-pose arm receiver ready."));
}

void loop() {
  static char line[128];
  static size_t lp = 0;
  unsigned long now = millis();

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[lp] = '\0';
      lp = 0;

      char work[128];
      strncpy(work, line, sizeof(work) - 1);
      work[sizeof(work) - 1] = '\0';

      float h[7];
      if (parse7(work, h)) {
        applyHandPose(h[1], h[2], h[3], h[4], clampf(h[5], 0.0f, 1.0f), h[6]);
        lastRxMs = now;
        // DEBUG: prove the Arduino received an H line and what it set servo 0 to.
        // Remove these prints once the arm is moving.
        Serial.print(F("RX H -> ch0="));
        Serial.print(outDeg[0]);
        Serial.print(F(" ch1="));
        Serial.print(outDeg[1]);
        Serial.print(F(" ch2="));
        Serial.print(outDeg[2]);
        Serial.print(F(" ch5="));
        Serial.println(outDeg[5]);
      } else {
        strncpy(work, line, sizeof(work) - 1);
        work[sizeof(work) - 1] = '\0';
        float n[6];
        if (parseNeutral(work, n)) {
          for (uint8_t i = 0; i < SERVO_COUNT; i++) neutralDeg[i] = clampf(n[i], 0.0f, 180.0f);
          stopAllServos();
          lastRxMs = now;
          Serial.println(F("Neutral updated."));
        } else {
          strncpy(work, line, sizeof(work) - 1);
          work[sizeof(work) - 1] = '\0';
          int id = -1;
          float pct = 0.0f;
          if (parseDirectSpeed(work, &id, &pct)) {
            // S id deg  — set channel `id` to absolute angle `deg` (0..180 degrees).
            // Conversion to HCPCA9685 0..450 units happens in posToHC().
            if (id >= 0 && id < SERVO_COUNT) {
              outDeg[id] = clampf(pct, POS_MIN[id], POS_MAX[id]);
              HCPCA9685.Servo(SERVO_CHANNELS[id], posToHC(outDeg[id]));
            }
            lastRxMs = now;
          } else {
            strncpy(work, line, sizeof(work) - 1);
            work[sizeof(work) - 1] = '\0';
            int onoff = 0;
            if (parseBaseEnable(work, &onoff)) {
              baseEnabled = (onoff != 0);
              Serial.print(F("Base enabled: "));
              Serial.println(baseEnabled ? F("1") : F("0"));
              lastRxMs = now;
            }
          }
        }
      }
    } else {
      if (lp < sizeof(line) - 1) line[lp++] = c;
      else lp = 0;
    }
  }

  if (lastRxMs != 0 && (now - lastRxMs > WATCHDOG_STOP_MS)) {
    stopAllServos();
    lastRxMs = 0;
  }
}
