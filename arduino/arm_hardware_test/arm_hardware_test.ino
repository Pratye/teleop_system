/**
 * Arm hardware bring-up test for CONTINUOUS-rotation MG996R servos.
 *
 * Commands:
 *   h               help
 *   i               I2C scan
 *   z               stop all servos (move to per-joint neutral stop)
 *   k               capture current outputs as new neutral stop
 *   s <id> <pct>    joint speed, pct in [-100..100]
 *   p <pct>         shoulder pair speed (J0/J1, mirrored)
 *   a <pct>         all joints speed (shoulder pair mirrored)
 *   w               sweep speed test
 *   e <0|1>         stepper disable/enable
 *   t <steps>       base stepper relative move
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

const unsigned long BAUD = 115200;
const uint8_t SERVO_COUNT = 6;
const uint8_t SERVO_CHANNELS[SERVO_COUNT] = {0, 1, 2, 3, 4, 5};
const char *JOINT_NAMES[SERVO_COUNT] = {
  "Shoulder-Y #1", "Shoulder-Y #2", "Elbow-Y", "Wrist-Y", "Wrist-Rotate-X", "Gripper"
};
const uint8_t PCA_ADDR = 0x40;
const int SERVO_FREQ_HZ = 50;
const int SERVO_MIN_US = 600;
const int SERVO_MAX_US = 2400;

// Continuous-servo tuning.
const float SPEED_SPAN_DEG = 24.0f;         // +/- command range around neutral
const float SHOULDER_PAIR_OFFSET_DEG = 0.0f;
float stopDeg[SERVO_COUNT] = {90, 90, 90, 90, 90, 90};  // neutral stop per channel
float outDeg[SERVO_COUNT] = {90, 90, 90, 90, 90, 90};   // last output command per channel

const uint8_t PIN_STEPPER_STEP = 5;
const uint8_t PIN_STEPPER_DIR = 4;
const uint8_t PIN_STEPPER_EN = 6;
const int STEP_PULSE_US = 600;
const int MAX_TEST_STEPS = 1000;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA_ADDR);

float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

uint16_t microsToPcaTicks(int micros) {
  long ticks = (long)micros * 4096L / 20000L;
  if (ticks < 0) ticks = 0;
  if (ticks > 4095) ticks = 4095;
  return (uint16_t)ticks;
}

uint16_t degToTicks(float deg) {
  deg = clampf(deg, 0.0f, 180.0f);
  float us = SERVO_MIN_US + (deg / 180.0f) * (float)(SERVO_MAX_US - SERVO_MIN_US);
  return microsToPcaTicks((int)(us + 0.5f));
}

void writeServoDeg(uint8_t id, float deg) {
  if (id >= SERVO_COUNT) return;
  outDeg[id] = clampf(deg, 0.0f, 180.0f);
  pwm.setPWM(SERVO_CHANNELS[id], 0, degToTicks(outDeg[id]));
}

void writeSpeedSingle(uint8_t id, float speed01) {
  float cmd = stopDeg[id] + clampf(speed01, -1.0f, 1.0f) * SPEED_SPAN_DEG;
  writeServoDeg(id, cmd);
}

void writeShoulderPairSpeed(float speed01) {
  speed01 = clampf(speed01, -1.0f, 1.0f);
  // Pair is mirrored mechanically.
  writeServoDeg(0, stopDeg[0] + speed01 * SPEED_SPAN_DEG);
  writeServoDeg(1, stopDeg[1] - speed01 * SPEED_SPAN_DEG + SHOULDER_PAIR_OFFSET_DEG);
}

void stopAllServos() {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) writeServoDeg(i, stopDeg[i]);
}

void captureCurrentAsStop() {
  for (uint8_t i = 0; i < SERVO_COUNT; i++) stopDeg[i] = outDeg[i];
  Serial.println(F("Captured current outputs as new neutral stop."));
}

void printStopPose() {
  Serial.print(F("Stop pose: "));
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    if (i) Serial.print(F(", "));
    Serial.print((int)stopDeg[i]);
  }
  Serial.println();
}

void stepperEnable(bool enable) {
  digitalWrite(PIN_STEPPER_EN, enable ? LOW : HIGH);
}

void stepRelative(int steps) {
  if (steps == 0) return;
  steps = (int)clampf((float)steps, -MAX_TEST_STEPS, MAX_TEST_STEPS);
  digitalWrite(PIN_STEPPER_DIR, (steps > 0) ? HIGH : LOW);
  int n = (steps > 0) ? steps : -steps;
  for (int i = 0; i < n; i++) {
    digitalWrite(PIN_STEPPER_STEP, HIGH);
    delayMicroseconds(STEP_PULSE_US);
    digitalWrite(PIN_STEPPER_STEP, LOW);
    delayMicroseconds(STEP_PULSE_US);
  }
}

bool i2cDevicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return (Wire.endTransmission() == 0);
}

void printHelp() {
  Serial.println(F("\n--- Continuous Servo Test ---"));
  Serial.println(F("h               : help"));
  Serial.println(F("i               : I2C scan + PCA9685 check"));
  Serial.println(F("z               : stop all servos (neutral stop)"));
  Serial.println(F("k               : capture current outputs as neutral stop"));
  Serial.println(F("s <id> <pct>    : set joint speed -100..100"));
  Serial.println(F("p <pct>         : shoulder pair speed -100..100"));
  Serial.println(F("a <pct>         : all joints speed -100..100"));
  Serial.println(F("w               : speed sweep test"));
  Serial.println(F("e <0|1>         : stepper disable/enable"));
  Serial.println(F("t <steps>       : base stepper relative move"));
}

void printI2cStatus() {
  Serial.println(F("I2C scan:"));
  bool foundAny = false;
  for (uint8_t addr = 1; addr < 127; addr++) {
    if (i2cDevicePresent(addr)) {
      foundAny = true;
      Serial.print(F("  Found device at 0x"));
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
    }
  }
  if (!foundAny) Serial.println(F("  No I2C device found"));
  Serial.print(F("PCA9685 @0x40: "));
  Serial.println(i2cDevicePresent(PCA_ADDR) ? F("OK") : F("NOT FOUND"));
}

void speedSweep() {
  Serial.println(F("Speed sweep test..."));
  for (uint8_t i = 0; i < SERVO_COUNT; i++) {
    if (i == 0 || i == 1) continue;  // shoulder tested as pair
    Serial.print(F("Joint ")); Serial.print(i); Serial.print(F(" ")); Serial.println(JOINT_NAMES[i]);
    writeSpeedSingle(i, 0.5f); delay(500);
    writeSpeedSingle(i, -0.5f); delay(500);
    writeSpeedSingle(i, 0.0f); delay(250);
  }
  writeShoulderPairSpeed(0.5f); delay(500);
  writeShoulderPairSpeed(-0.5f); delay(500);
  writeShoulderPairSpeed(0.0f);
  Serial.println(F("Speed sweep done."));
}

static void handleCommandLine(char *work) {
  char *tok = strtok(work, " \t\r\n,");
  if (!tok || !tok[0]) return;
  char cmd = (char)tolower((unsigned char)tok[0]);

  switch (cmd) {
    case 'h': printHelp(); break;
    case 'i': printI2cStatus(); break;
    case 'z': stopAllServos(); printStopPose(); break;
    case 'k': captureCurrentAsStop(); printStopPose(); break;
    case 's': {
      char *tid = strtok(NULL, " \t\r\n,");
      char *tpct = strtok(NULL, " \t\r\n,");
      if (!tid || !tpct) { Serial.println(F("Usage: s <id> <pct -100..100>")); break; }
      int id = atoi(tid);
      float spd = clampf((float)atof(tpct) / 100.0f, -1.0f, 1.0f);
      if (id < 0 || id >= SERVO_COUNT) { Serial.println(F("id must be 0..5")); break; }
      if (id == 0 || id == 1) {
        writeShoulderPairSpeed(spd);
        Serial.print(F("Shoulder pair speed: ")); Serial.println((int)(spd * 100.0f));
      } else {
        writeSpeedSingle((uint8_t)id, spd);
        Serial.print(F("Joint ")); Serial.print(id); Serial.print(F(" speed: "));
        Serial.println((int)(spd * 100.0f));
      }
      break;
    }
    case 'p': {
      char *tpct = strtok(NULL, " \t\r\n,");
      if (!tpct) { Serial.println(F("Usage: p <pct -100..100>")); break; }
      float spd = clampf((float)atof(tpct) / 100.0f, -1.0f, 1.0f);
      writeShoulderPairSpeed(spd);
      break;
    }
    case 'a': {
      char *tpct = strtok(NULL, " \t\r\n,");
      if (!tpct) { Serial.println(F("Usage: a <pct -100..100>")); break; }
      float spd = clampf((float)atof(tpct) / 100.0f, -1.0f, 1.0f);
      writeShoulderPairSpeed(spd);
      for (uint8_t i = 2; i < SERVO_COUNT; i++) writeSpeedSingle(i, spd);
      break;
    }
    case 'w': speedSweep(); break;
    case 'e': {
      char *ten = strtok(NULL, " \t\r\n,");
      if (!ten) { Serial.println(F("Usage: e <0|1>")); break; }
      stepperEnable(atoi(ten) != 0);
      break;
    }
    case 't': {
      char *tst = strtok(NULL, " \t\r\n,");
      if (!tst) { Serial.println(F("Usage: t <steps>")); break; }
      stepRelative(atoi(tst));
      break;
    }
    default:
      Serial.println(F("Unknown cmd. Type h"));
      break;
  }
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin();
  pinMode(PIN_STEPPER_STEP, OUTPUT);
  pinMode(PIN_STEPPER_DIR, OUTPUT);
  pinMode(PIN_STEPPER_EN, OUTPUT);
  stepperEnable(true);
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ_HZ);
  delay(10);
  stopAllServos();
  Serial.println(F("\nContinuous-servo hardware test ready."));
  printStopPose();
  printI2cStatus();
  printHelp();
}

void loop() {
  if (!Serial.available()) return;
  static char line[64];
  static size_t lp = 0;

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      if (lp == 0) continue;
      line[lp] = '\0';
      lp = 0;
      break;
    }
    if (c == ',') c = ' ';
    if (lp < sizeof(line) - 1) line[lp++] = c;
    else lp = 0;
  }

  if (line[0] == '\0') return;
  static char execLine[64];
  strncpy(execLine, line, sizeof(execLine) - 1);
  execLine[sizeof(execLine) - 1] = '\0';
  handleCommandLine(execLine);
  line[0] = '\0';
}
