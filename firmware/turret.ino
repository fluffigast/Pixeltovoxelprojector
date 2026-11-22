// turret.ino — galvo-aimed laser turret, driven by the browser WASM
// detector over USB-serial (WebSerial).
//
// Hardware assumed:
//   - Raspberry Pi Pico 2 (arduino-pico core) / Teensy 4.x / Uno-class MCU
//   - MCP4822 dual 12-bit DAC on SPI (CS pin below)
//     -> DAC_A -> galvo X-axis driver input
//     -> DAC_B -> galvo Y-axis driver input
//     NOTE: MCP4822 outputs 0..4.096 V single-ended. Most galvo drivers
//     want +/-5 V or +/-10 V differential, so put a dual-opamp level
//     shifter between the DAC and the driver (e.g. OPA2134 with
//     +/-12 V rails, gain ~2.44, bias to center on 2.048 V).
//   - Laser TTL enable (active-HIGH) wired to LASER_TTL_PIN.
//     ALSO wire the laser driver's enable through the physical arming
//     switch in series — software must not be the only interlock.
//   - Physical arming key switch -> ARM_PIN, closes to GND = ARMED.
//     (Fail-safe: a broken wire reads HIGH via pull-up = disarmed.)
//   - Low-power guide laser (class 2, <1 mW, coaxial with the main beam
//     via a beam combiner) on AIM_PIN, used only for calibration.
//   - Enclosure lid microswitch -> LID_PIN (LOW = lid closed).
//   - Status LED on LED_PIN (optional).
//
// Protocol (text, newline-terminated, 115200 baud):
//   A <x> <y> <f>\n     aim at pixel (x,y); f=1 requests fire
//   H <h00..h22>\n      set 3x3 homography (9 floats, row-major)
//   K <x0> <y0> <x1> <y1>\n  set kill-zone rectangle in pixel coords
//   D <ms>\n            set max continuous dwell in ms
//   C <dx> <dy> <g>\n   calibration: raw DAC aim, guide laser g=0/1,
//                       main laser forced OFF (guide auto-off after 1 s)
//   V\n                 version/status request
// MCU responses:
//   OK\n, ERR <msg>\n, ST armed=<0|1> lid=<0|1> fire=<0|1>\n
//
// Safety model (do not weaken):
//   1. ARM_PIN must read LOW (hardware key switch closed to GND).
//   2. LID_PIN must read LOW (enclosure closed).
//   3. Target pixel must be inside the configured kill-zone rectangle.
//   4. Watchdog: no fresh A command in WATCHDOG_MS -> fire off.
//   5. Dwell limit: fire pin HIGH for more than maxDwellMs -> cutoff
//      until the next 'A ... 0' message resets it.

#include <SPI.h>

// ---- Pins ----
#if defined(ARDUINO_ARCH_RP2040)
// Pico / Pico 2: SPI0 default SCK=GP18, MOSI=GP19 (MISO GP16 unused)
static constexpr int DAC_CS_PIN     = 17;
static constexpr int LASER_TTL_PIN  = 15;
static constexpr int AIM_PIN        = 14;  // guide laser enable (active HIGH)
static constexpr int ARM_PIN        = 13;  // INPUT_PULLUP; key switch to GND = ARMED
static constexpr int LID_PIN        = 12;  // INPUT_PULLUP; switch closes to GND when lid is shut
#else
static constexpr int DAC_CS_PIN     = 10;
static constexpr int LASER_TTL_PIN  = 6;
static constexpr int AIM_PIN        = 7;   // guide laser enable (active HIGH)
static constexpr int ARM_PIN        = 2;   // INPUT_PULLUP; key switch to GND = ARMED
static constexpr int LID_PIN        = 3;   // INPUT_PULLUP; switch closes to GND when lid is shut
#endif
static constexpr int LED_PIN        = LED_BUILTIN;

// ---- Timing (ms) ----
static constexpr uint32_t WATCHDOG_MS       = 150;
static constexpr uint32_t DEFAULT_DWELL_MS  = 400;
static constexpr uint32_t STATUS_PERIOD_MS  = 250;
static constexpr uint32_t GUIDE_TIMEOUT_MS  = 1000;

// ---- DAC ----
static constexpr uint16_t DAC_MIN = 0;
static constexpr uint16_t DAC_MAX = 4095;
static constexpr uint16_t DAC_MID = 2048;

// ---- Calibration ----
// Row-major 3x3 homography mapping pixel (px, py) -> DAC (dac_x, dac_y).
// Defaults: identity + offset so a 1920x1080-ish stream centers mid-DAC.
// Overwrite at runtime with the 'H ...' command.
float H[9] = {
   1.0f,  0.0f, 0.0f,
   0.0f,  1.0f, 0.0f,
   0.0f,  0.0f, 1.0f
};

// Kill-zone rectangle in detector pixel space. Firing is refused outside.
// Default: a safe 10x10 pixel area at (0,0) that effectively disables
// firing until the host sends a real zone with 'K ...'.
float kzX0 = 0, kzY0 = 0, kzX1 = 10, kzY1 = 10;

uint32_t maxDwellMs = DEFAULT_DWELL_MS;

// ---- State ----
uint32_t lastAimMs      = 0;    // last time we got a valid A command
uint32_t fireStartMs    = 0;    // when the fire pin went HIGH
bool     fireActive     = false;
bool     dwellLatched   = false; // dwell cutoff until next '0' message
uint16_t curDacX        = DAC_MID;
uint16_t curDacY        = DAC_MID;
uint32_t lastStatusMs   = 0;
uint32_t lastGuideMs    = 0;
bool     guideOn        = false;

char lineBuf[192];
uint8_t lineLen = 0;

// ----------------------------------------------------------------------
// MCP4822 write. Channel A = X, B = Y. Gain=1 (0..2.048 V) OR Gain=0 for
// 2x range (0..4.096 V). We pick 2x so the opamp shifter has more swing.
// ----------------------------------------------------------------------
void dacWrite(uint16_t x, uint16_t y) {
  uint16_t cmdA = (0b0001 << 12) | (x & 0x0FFF); // chan A, buf=0, gain=1(2x), shdn=1
  uint16_t cmdB = (0b1001 << 12) | (y & 0x0FFF); // chan B, same flags
  digitalWrite(DAC_CS_PIN, LOW);
  SPI.transfer16(cmdA);
  digitalWrite(DAC_CS_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(DAC_CS_PIN, LOW);
  SPI.transfer16(cmdB);
  digitalWrite(DAC_CS_PIN, HIGH);
  curDacX = x; curDacY = y;
}

inline uint16_t clampDac(float v) {
  if (v < DAC_MIN) return DAC_MIN;
  if (v > DAC_MAX) return DAC_MAX;
  return (uint16_t)v;
}

bool pixelToDac(float px, float py, uint16_t& outX, uint16_t& outY) {
  float x = H[0]*px + H[1]*py + H[2];
  float y = H[3]*px + H[4]*py + H[5];
  float w = H[6]*px + H[7]*py + H[8];
  if (w == 0.0f) return false;
  outX = clampDac(x / w);
  outY = clampDac(y / w);
  return true;
}

bool insideKillZone(float px, float py) {
  return px >= kzX0 && px <= kzX1 && py >= kzY0 && py <= kzY1;
}

bool armed()    { return digitalRead(ARM_PIN) == LOW;  } // fail-safe: open wire = disarmed
bool lidClosed(){ return digitalRead(LID_PIN) == LOW;  }

void setFire(bool on) {
  if (on == fireActive) return;
  fireActive = on;
  digitalWrite(LASER_TTL_PIN, on ? HIGH : LOW);
  if (on) fireStartMs = millis();
}

void setGuide(bool on) {
  guideOn = on;
  digitalWrite(AIM_PIN, on ? HIGH : LOW);
  if (on) lastGuideMs = millis();
}

void disableAll() {
  setFire(false);
  setGuide(false);
  dwellLatched = false;
}

// ----------------------------------------------------------------------
// Line parsing. One command per line, newline-terminated.
// ----------------------------------------------------------------------
void handleA(char* args) {
  // "A <x> <y> <f>"
  float px = atof(strtok(args, " \t"));
  char* ts = strtok(NULL, " \t"); if (!ts) { Serial.println("ERR fmt"); return; }
  float py = atof(ts);
  char* fs = strtok(NULL, " \t"); if (!fs) { Serial.println("ERR fmt"); return; }
  int   f  = atoi(fs);

  uint16_t dx, dy;
  if (!pixelToDac(px, py, dx, dy)) { Serial.println("ERR homog"); return; }
  dacWrite(dx, dy);
  if (guideOn) setGuide(false);
  lastAimMs = millis();

  bool allowFire = (f != 0) && armed() && lidClosed() && insideKillZone(px, py) && !dwellLatched;
  if (f == 0) dwellLatched = false; // host explicitly released -> rearm
  setFire(allowFire);
  Serial.println("OK");
}

void handleH(char* args) {
  float v[9]; int n = 0;
  for (char* tok = strtok(args, " \t"); tok && n < 9; tok = strtok(NULL, " \t")) v[n++] = atof(tok);
  if (n != 9) { Serial.println("ERR H needs 9"); return; }
  for (int i = 0; i < 9; i++) H[i] = v[i];
  Serial.println("OK");
}

void handleK(char* args) {
  char* a = strtok(args, " \t"); if (!a) { Serial.println("ERR fmt"); return; }
  char* b = strtok(NULL, " \t"); if (!b) { Serial.println("ERR fmt"); return; }
  char* c = strtok(NULL, " \t"); if (!c) { Serial.println("ERR fmt"); return; }
  char* d = strtok(NULL, " \t"); if (!d) { Serial.println("ERR fmt"); return; }
  float x0 = atof(a), y0 = atof(b), x1 = atof(c), y1 = atof(d);
  kzX0 = fminf(x0, x1); kzX1 = fmaxf(x0, x1);
  kzY0 = fminf(y0, y1); kzY1 = fmaxf(y0, y1);
  Serial.println("OK");
}

void handleD(char* args) {
  long ms = atol(strtok(args, " \t"));
  if (ms < 10 || ms > 5000) { Serial.println("ERR range"); return; }
  maxDwellMs = (uint32_t)ms;
  Serial.println("OK");
}

void handleC(char* args) {
  // "C <dacx> <dacy> <guide>" — calibration only, never fires the main laser.
  char* a = strtok(args, " \t"); char* b = strtok(NULL, " \t"); char* c = strtok(NULL, " \t");
  if (!a || !b || !c) { Serial.println("ERR fmt"); return; }
  setFire(false);
  dwellLatched = false;
  dacWrite(clampDac(atof(a)), clampDac(atof(b)));
  setGuide(atoi(c) != 0);
  Serial.println("OK");
}

void handleV() {
  Serial.print("ST armed="); Serial.print(armed() ? 1 : 0);
  Serial.print(" lid=");     Serial.print(lidClosed() ? 1 : 0);
  Serial.print(" fire=");    Serial.print(fireActive ? 1 : 0);
  Serial.print(" dwell=");   Serial.println(maxDwellMs);
}

void dispatch(char* line) {
  while (*line == ' ' || *line == '\t') line++;
  char op = *line;
  if (!op) return;
  char* args = line + 1;
  while (*args == ' ' || *args == '\t') args++;
  switch (op) {
    case 'A': handleA(args); break;
    case 'H': handleH(args); break;
    case 'K': handleK(args); break;
    case 'D': handleD(args); break;
    case 'C': handleC(args); break;
    case 'V': handleV();     break;
    default:  Serial.println("ERR op");
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = 0;
      if (lineLen > 0) dispatch(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0; // overflow -> drop
    }
  }
}

// ----------------------------------------------------------------------
void setup() {
  pinMode(DAC_CS_PIN,    OUTPUT); digitalWrite(DAC_CS_PIN, HIGH);
  pinMode(LASER_TTL_PIN, OUTPUT); digitalWrite(LASER_TTL_PIN, LOW);
  pinMode(AIM_PIN,       OUTPUT); digitalWrite(AIM_PIN, LOW);
  pinMode(ARM_PIN,       INPUT_PULLUP);
  pinMode(LID_PIN,       INPUT_PULLUP);
  pinMode(LED_PIN,       OUTPUT);
  SPI.begin();
  SPI.beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE0));
  dacWrite(DAC_MID, DAC_MID);
  Serial.begin(115200);
  // Wait up to 1 s for USB-serial on Teensy; non-blocking on Uno.
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 1000) {}
  Serial.println("ST turret boot");
}

void loop() {
  readSerial();

  const uint32_t now = millis();

  // Watchdog: no fresh aim command recently -> fire off.
  if (fireActive && (now - lastAimMs) > WATCHDOG_MS) {
    setFire(false);
  }
  // Dwell cutoff: continuous fire too long -> latch off until next release.
  if (fireActive && (now - fireStartMs) > maxDwellMs) {
    setFire(false);
    dwellLatched = true;
  }
  // Guide laser auto-off if the host stops refreshing C commands.
  if (guideOn && (now - lastGuideMs) > GUIDE_TIMEOUT_MS) setGuide(false);
  // Interlock re-check every loop; if ARM/LID changes, drop fire immediately.
  if (fireActive && (!armed() || !lidClosed())) {
    setFire(false);
  }

  // Heartbeat LED mirrors armed status.
  digitalWrite(LED_PIN, armed() && lidClosed() ? HIGH : LOW);

  // Periodic status telemetry for the host UI.
  if (now - lastStatusMs > STATUS_PERIOD_MS) {
    lastStatusMs = now;
    Serial.print("ST armed="); Serial.print(armed() ? 1 : 0);
    Serial.print(" lid=");     Serial.print(lidClosed() ? 1 : 0);
    Serial.print(" fire=");    Serial.print(fireActive ? 1 : 0);
    Serial.print(" dwelllatched="); Serial.println(dwellLatched ? 1 : 0);
  }
}
