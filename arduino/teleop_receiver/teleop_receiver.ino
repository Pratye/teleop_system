/**
 * Teleop receiver for:
 * - Arduino Uno + HC-05 serial link on hardware UART (D0/D1)
 * - 6x MG996R continuous-rotation servos on PCA9685 (channels 0..5)
 * - NEMA17 stepper on A4988 (DIR/STEP/ENABLE pins)
 *
 * Wiring notes for Arduino Uno:
 * - I2C to PCA9685: SCL=A5, SDA=A4
 * - HC-05: RX -> Arduino TX (D1), TX -> Arduino RX (D0)
 *
 * Line protocol @ 115200 baud (camera teleop sender compatible):
 *   B SH EL WY WX G
 * where:
 * - B  = base revolute Z command (degrees, absolute) -> A4988 stepper
 * For continuous servos, incoming values are interpreted as pseudo-angle
 * targets and converted to speed commands around per-joint neutral stops.
 * - SH = shoulder pseudo-angle (around SH_NEUTRAL_DEG)
 * - EL = elbow pseudo-angle (around EL_NEUTRAL_DEG)
 * - WY = wrist-Y pseudo-angle (around WY_NEUTRAL_DEG)
 * - WX = wrist-X pseudo-angle (around WX_NEUTRAL_DEG)
 * - G  = gripper open scalar [0..1] -> mapped to speed [-1..1]
 *
 * Incoming lines are parsed with strtok/atof (AVR sscanf %f is not reliable).
 * This sketch intentionally contains no glove sensor mapping;
 * it only accepts direct robotic-arm joint commands.
 */

 #include <Wire.h>
 #include <Adafruit_PWMServoDriver.h>
 #include <stdlib.h>
 #include <string.h>
 
 const unsigned long BAUD = 115200;
 const unsigned long WATCHDOG_MS = 600;
 
 const uint8_t SERVO_COUNT = 6;
 const uint8_t SERVO_CHANNELS[SERVO_COUNT] = {0, 1, 2, 3, 4, 5};
 
 // MG966R practical pulse range (microseconds). Tune per joint if needed.
 const int SERVO_MIN_US = 600;
 const int SERVO_MAX_US = 2400;
 const int SERVO_FREQ_HZ = 50;
 
 // A4988 pins (based on your wiring)
 const uint8_t PIN_STEPPER_STEP = 5;
 const uint8_t PIN_STEPPER_DIR = 4;
 const uint8_t PIN_STEPPER_EN = 6;  // LOW = enabled on most A4988 boards
 
 const int MAX_STEPS_PER_CMD = 200;
 const int STEP_PULSE_US = 600;
const float BASE_STEPS_PER_DEG = 8.8889f;  // 1/16 microstep on 1.8 deg motor (~3200 steps/rev)
const int BASE_MAX_STEP_DELTA = 120;

// Continuous-servo tuning (neutral stop + authority around stop).
const float SHOULDER_PAIR_OFFSET_DEG = 0.0f;
const float SPEED_SPAN_DEG = 24.0f;    // output command span around stop
const float TARGET_SPAN_DEG = 45.0f;   // pseudo-angle delta that maps to full speed
const float WATCHDOG_STOP_MS = 700;

const float SH_NEUTRAL_DEG = 90.0f;
const float EL_NEUTRAL_DEG = 90.0f;
const float WY_NEUTRAL_DEG = 90.0f;
const float WX_NEUTRAL_DEG = 90.0f;
const float GR_NEUTRAL_DEG = 90.0f;
 
 Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
float stopDeg[SERVO_COUNT] = {SH_NEUTRAL_DEG, 180.0f - SH_NEUTRAL_DEG + SHOULDER_PAIR_OFFSET_DEG, EL_NEUTRAL_DEG, WY_NEUTRAL_DEG, WX_NEUTRAL_DEG, GR_NEUTRAL_DEG};
float outDeg[SERVO_COUNT] = {SH_NEUTRAL_DEG, 180.0f - SH_NEUTRAL_DEG + SHOULDER_PAIR_OFFSET_DEG, EL_NEUTRAL_DEG, WY_NEUTRAL_DEG, WX_NEUTRAL_DEG, GR_NEUTRAL_DEG};
 unsigned long lastRxMs = 0;
float lastBaseDeg = 0.0f;
bool hasBaseRef = false;
 
 float clampf(float v, float lo, float hi) {
   if (v < lo) return lo;
   if (v > hi) return hi;
   return v;
 }

// AVR libc sscanf does not reliably support %f — use strtok + atof.
static bool parseSixFloats(char *buf, float out[6]) {
  char *tok = strtok(buf, " \t,");
  for (int i = 0; i < 6; i++) {
    if (!tok) return false;
    out[i] = (float)atof(tok);
    tok = strtok(NULL, " \t,");
  }
  return true;
}
 
 uint16_t microsToPcaTicks(int micros) {
   // 20 ms frame at 50Hz -> convert microseconds to 12-bit ticks.
   long ticks = (long)micros * 4096L / 20000L;
   if (ticks < 0) ticks = 0;
   if (ticks > 4095) ticks = 4095;
   return (uint16_t)ticks;
 }
 
 uint16_t degToTicks(float deg) {
   deg = clampf(deg, 0.0f, 180.0f);
   float u = SERVO_MIN_US + (deg / 180.0f) * float(SERVO_MAX_US - SERVO_MIN_US);
   return microsToPcaTicks((int)(u + 0.5f));
 }

void applyServos(const float joints[SERVO_COUNT]) {
   for (uint8_t i = 0; i < SERVO_COUNT; i++) {
     pwm.setPWM(SERVO_CHANNELS[i], 0, degToTicks(joints[i]));
   }
 }
 
 void stepRelative(int steps) {
   if (steps == 0) return;
   steps = (int)clampf((float)steps, -MAX_STEPS_PER_CMD, MAX_STEPS_PER_CMD);
 
   digitalWrite(PIN_STEPPER_DIR, (steps > 0) ? HIGH : LOW);
   int absSteps = (steps > 0) ? steps : -steps;
   for (int i = 0; i < absSteps; i++) {
     digitalWrite(PIN_STEPPER_STEP, HIGH);
     delayMicroseconds(STEP_PULSE_US);
     digitalWrite(PIN_STEPPER_STEP, LOW);
     delayMicroseconds(STEP_PULSE_US);
   }
 }

int baseDegToStepDelta(float baseDegAbs) {
  // Sender provides absolute base angle; convert to relative step pulses.
  if (!hasBaseRef) {
    lastBaseDeg = baseDegAbs;
    hasBaseRef = true;
    return 0;
  }
  float dDeg = baseDegAbs - lastBaseDeg;
  lastBaseDeg = baseDegAbs;
  int dSteps = (int)(dDeg * BASE_STEPS_PER_DEG + ((dDeg >= 0.0f) ? 0.5f : -0.5f));
  if (dSteps > BASE_MAX_STEP_DELTA) dSteps = BASE_MAX_STEP_DELTA;
  if (dSteps < -BASE_MAX_STEP_DELTA) dSteps = -BASE_MAX_STEP_DELTA;
  return dSteps;
}

float speedFromTarget(float target, float neutral) {
  return clampf((target - neutral) / TARGET_SPAN_DEG, -1.0f, 1.0f);
}

void writeContinuousSingle(uint8_t id, float speed01) {
  outDeg[id] = clampf(stopDeg[id] + clampf(speed01, -1.0f, 1.0f) * SPEED_SPAN_DEG, 0.0f, 180.0f);
}

void applyContinuousJoints(float shTgt, float elTgt, float wyTgt, float wxTgt, float gripOpen01) {
  float shSpd = speedFromTarget(shTgt, SH_NEUTRAL_DEG);
  float elSpd = speedFromTarget(elTgt, EL_NEUTRAL_DEG);
  float wySpd = speedFromTarget(wyTgt, WY_NEUTRAL_DEG);
  float wxSpd = speedFromTarget(wxTgt, WX_NEUTRAL_DEG);
  float grSpd = clampf((gripOpen01 * 2.0f) - 1.0f, -1.0f, 1.0f);

  // Shoulder pair mirrored.
  outDeg[0] = clampf(stopDeg[0] + shSpd * SPEED_SPAN_DEG, 0.0f, 180.0f);
  outDeg[1] = clampf(stopDeg[1] - shSpd * SPEED_SPAN_DEG + SHOULDER_PAIR_OFFSET_DEG, 0.0f, 180.0f);
  writeContinuousSingle(2, elSpd);
  writeContinuousSingle(3, wySpd);
  writeContinuousSingle(4, wxSpd);
  writeContinuousSingle(5, grSpd);
  applyServos(outDeg);
}

void stopAllContinuous() {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) outDeg[i] = stopDeg[i];
  applyServos(outDeg);
}
 
 void setup() {
   Serial.begin(BAUD);
 
   Wire.begin();
   pwm.begin();
   pwm.setPWMFreq(SERVO_FREQ_HZ);
   delay(10);
 
   pinMode(PIN_STEPPER_STEP, OUTPUT);
   pinMode(PIN_STEPPER_DIR, OUTPUT);
   pinMode(PIN_STEPPER_EN, OUTPUT);
   digitalWrite(PIN_STEPPER_EN, LOW);
 
  stopAllContinuous();
 }
 
 void loop() {
   unsigned long now = millis();
   static char lineBuf[128];
   static size_t lp = 0;
 
   while (Serial.available()) {
     char c = (char)Serial.read();
     lastRxMs = now;
     if (c == '\r') continue;
     if (c == '\n') {
       lineBuf[lp] = '\0';
       lp = 0;
 
      float v[6];
      if (parseSixFloats(lineBuf, v)) {
        float baseDeg = v[0];
        float shDeg = v[1];
        float elDeg = v[2];
        float wyDeg = v[3];
        float wxDeg = v[4];
        float gripOpen = v[5];

        int stepDelta = baseDegToStepDelta(baseDeg);

        gripOpen = clampf(gripOpen, 0.0f, 1.0f);
        applyContinuousJoints(shDeg, elDeg, wyDeg, wxDeg, gripOpen);
        stepRelative(stepDelta);
      }
     } else {
       if (lp < sizeof(lineBuf) - 1) {
         lineBuf[lp++] = c;
       } else {
         lp = 0;
       }
     }
   }
 
  if (now - lastRxMs > WATCHDOG_STOP_MS && lastRxMs != 0) {
    // Stop continuous servos when link is idle.
    stopAllContinuous();
   }
 }
 