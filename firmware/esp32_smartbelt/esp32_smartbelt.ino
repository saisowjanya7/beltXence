/*
 * SmartBelt v2 — ESP32 Firmware
 * Sensors: MLX90614 (temp, I2C 0x5A), MPU-6050 (vibration, I2C 0x68),
 *          HX711 + load cell (tension), E18-D80NK IR proximity (belt speed)
 * Output:  Single-line JSON over Serial at 115200 baud, 5 Hz
 *
 * ─── WIRING ──────────────────────────────────────────────────────────────────
 * MLX90614   SDA -> GPIO 21 | SCL -> GPIO 22 | VCC -> 3.3V | GND -> GND
 * MPU-6050   SDA -> GPIO 21 | SCL -> GPIO 22 | VCC -> 3.3V | GND -> GND | AD0 -> GND
 * HX711      DOUT -> GPIO 16 | SCK -> GPIO 17 | VCC -> 3.3V/5V | GND -> GND
 * E18-D80NK  OUT -> GPIO 34 | VCC -> 5V | GND -> GND (NPN-NO, active LOW)
 *
 * ─── CALIBRATION CONSTANTS ───────────────────────────────────────────────────
 * HX711_CALIBRATION_FACTOR  : Raw counts per kg.
 *   To calibrate: set factor=1, place known weight, note raw value, factor = raw/weight_kg
 * BELT_MARKERS               : Number of reflective tape strips on the belt (default 1)
 * BELT_CIRCUMFERENCE_M       : Total belt length in metres
 * IR_DEBOUNCE_MS             : Minimum ms between valid IR pulses
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <MPU6050.h>
#include <HX711.h>
#include <ArduinoJson.h>

// ── Calibration constants ──
#define HX711_CALIBRATION_FACTOR  2280.0f   // raw counts per kg (calibrate with known weight)
#define HX711_ZERO_OFFSET         0.0f      // tare offset in kg
#define BELT_MARKERS              1         // number of IR markers on belt
#define BELT_CIRCUMFERENCE_M      2.0f      // belt total length in metres
#define IR_DEBOUNCE_MS            50        // min ms between valid pulses
#define SAMPLE_INTERVAL_MS        200       // emit JSON every 200 ms (5 Hz)

// ── Pin assignments ──────────────────────────────────────────────────────────
#define HX711_DOUT_PIN   16
#define HX711_SCK_PIN    17
#define IR_SENSOR_PIN    34

Adafruit_MLX90614 mlx;
MPU6050 mpu;
HX711 scale;

// IR pulse counting (ISR)
volatile uint32_t irPulseCount = 0;
volatile uint32_t lastIrTriggerMs = 0;

void IRAM_ATTR onIrPulse() {
  uint32_t now = millis();
  if (now - lastIrTriggerMs >= IR_DEBOUNCE_MS) {
    irPulseCount++;
    lastIrTriggerMs = now;
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);  // SDA=21, SCL=22

  // MLX90614
  if (!mlx.begin()) {
    Serial.println("{\"error\":\"MLX90614 not found\"}");
  }

  // MPU-6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("{\"error\":\"MPU6050 not found\"}");
  }

  // HX711
  scale.begin(HX711_DOUT_PIN, HX711_SCK_PIN);
  scale.set_scale(HX711_CALIBRATION_FACTOR);
  scale.tare();  // zero on boot

  // IR sensor
  pinMode(IR_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IR_SENSOR_PIN), onIrPulse, FALLING);
}

float computeVibRms(int16_t ax, int16_t ay, int16_t az) {
  // MPU6050 raw -> g (+-2g range -> divide by 16384)
  float gx = ax / 16384.0f;
  float gy = ay / 16384.0f;
  float gz = az / 16384.0f;
  // Remove gravity component from Z (assume sensor mounted flat)
  gz = gz - 1.0f;
  return sqrt(gx * gx + gy * gy + gz * gz);
}

void loop() {
  static uint32_t lastSampleMs = 0;
  static uint32_t lastPulseCountSnapshot = 0;
  static uint32_t lastSpeedCalcMs = 0;
  static float beltSpeedMps = 0.0f;

  uint32_t now = millis();

  // Compute belt speed from IR pulse delta every SAMPLE_INTERVAL_MS
  if (now - lastSpeedCalcMs >= SAMPLE_INTERVAL_MS) {
    noInterrupts();
    uint32_t pulsesDelta = irPulseCount - lastPulseCountSnapshot;
    lastPulseCountSnapshot = irPulseCount;
    interrupts();

    float elapsedSec = (now - lastSpeedCalcMs) / 1000.0f;
    if (elapsedSec > 0.0f) {
      beltSpeedMps = (pulsesDelta / (float)BELT_MARKERS) * BELT_CIRCUMFERENCE_M / elapsedSec;
    }
    lastSpeedCalcMs = now;
  }

  // Emit JSON packet every SAMPLE_INTERVAL_MS
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    // ── Read MLX90614 ──
    float tempC = mlx.readObjectTempC();

    // ── Read MPU-6050 ──
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    float vibRms = computeVibRms(ax, ay, az);

    // ── Read HX711 ──
    float loadKg = 0.0f;
    if (scale.is_ready()) {
      loadKg = scale.get_units(1) - HX711_ZERO_OFFSET;
      if (loadKg < 0.0f) loadKg = 0.0f;
    }

    // ── Read IR (current state) ──
    int irState = digitalRead(IR_SENSOR_PIN);  // LOW = object detected

    // ── Emit JSON ──
    StaticJsonDocument<256> doc;
    doc["ts"]             = now;
    doc["temp_c"]         = serialized(String(tempC, 2));
    doc["vib_x"]          = serialized(String(ax / 16384.0f, 4));
    doc["vib_y"]          = serialized(String(ay / 16384.0f, 4));
    doc["vib_z"]          = serialized(String(az / 16384.0f, 4));
    doc["vib_rms"]        = serialized(String(vibRms, 4));
    doc["load_kg"]        = serialized(String(loadKg, 3));
    doc["ir_state"]       = irState;           // 0=object present, 1=clear
    doc["belt_speed_mps"] = serialized(String(beltSpeedMps, 3));

    serializeJson(doc, Serial);
    Serial.println();
  }
}
