// =============================================================================
// ESP32 - Weigh Station & Kinematics Telemetry Node (COM21)
// Hardware:
//   - ESP32 Development Board
//   - MPU-6050 6-Axis MotionTracking Sensor (Accelerometer + Gyroscope + Temp)
//   - HX711 24-bit ADC Load Cell Weighing Module (Strain Gauge)
//
// Pinout Wiring:
//   MPU-6050:
//     * VCC  -> ESP32 3.3V
//     * GND  -> ESP32 GND
//     * SDA  -> ESP32 GPIO 21 (I2C SDA)
//     * SCL  -> ESP32 GPIO 22 (I2C SCL)
//     * AD0  -> GND (I2C address 0x68)
//
//   HX711 Load Cell Module:
//     * VCC  -> ESP32 5V (VIN) or 3.3V
//     * GND  -> ESP32 GND
//     * DOUT -> ESP32 GPIO 18 (Data)
//     * SCK  -> ESP32 GPIO 19 (Clock)
//
// Features:
//   - Pure Wire.h implementation (Zero external library dependencies)
//   - Non-blocking HX711 driver with auto-timeout (safe if scale is disconnected)
//   - Real-time digital scale tare & calibration
//   - Complementary Filter (Alpha = 0.98) for Pitch, Roll & Yaw
//   - Fast 50 Hz Telemetry Stream @ 115200 Baud
//   - Remote TARE command support via Serial
// =============================================================================

#include <Wire.h>
#include <Arduino.h>

#define BOARD_ID 3

// Pin Definitions
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define I2C_FREQ    400000

#define HX711_DOUT  18
#define HX711_SCK   19

#define LED_PIN     2 // Onboard LED

// MPU-6050 Registers & Address
const uint8_t MPU_ADDR = 0x68;

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_TEMP_OUT_H   0x41
#define REG_GYRO_XOUT_H  0x43
#define REG_PWR_MGMT_1   0x6B
#define REG_WHO_AM_I     0x75

// Telemetry interval: 20 ms = 50 Hz
const unsigned long TELEMETRY_INTERVAL_MS = 20;

// Load cell calibration
float scale_factor = 2280.0f; // Counts per gram (calibrate with known weight)
long zero_offset   = 0;
float current_weight_g = 0.0f;
bool hx711_detected = false;

// Gyro calibration offsets
float gyro_x_offset = 0.0f;
float gyro_y_offset = 0.0f;
float gyro_z_offset = 0.0f;

// Filtered orientation angles
float pitch = 0.0f;
float roll  = 0.0f;
float yaw   = 0.0f;
const float ALPHA = 0.98f;

unsigned long prevTime = 0;
unsigned long lastTelemetryTime = 0;
unsigned long lastWeightReadTime = 0;
bool mpuConnected = false;

// =============================================================================
// HX711 Load Cell Driver (Pure GPIO Bit-Bang, Non-blocking)
// =============================================================================

bool hx711_is_ready() {
  return digitalRead(HX711_DOUT) == LOW;
}

long hx711_read_raw() {
  if (!hx711_is_ready()) return zero_offset;

  long count = 0;
  for (int i = 0; i < 24; i++) {
    digitalWrite(HX711_SCK, HIGH);
    delayMicroseconds(1);
    count = count << 1;
    digitalWrite(HX711_SCK, LOW);
    delayMicroseconds(1);
    if (digitalRead(HX711_DOUT) == HIGH) {
      count++;
    }
  }

  // 25th pulse sets channel A, gain 128 for next reading
  digitalWrite(HX711_SCK, HIGH);
  delayMicroseconds(1);
  digitalWrite(HX711_SCK, LOW);
  delayMicroseconds(1);

  // Sign extension for 24-bit two's complement
  if (count & 0x800000) {
    count |= 0xFF000000;
  }
  return count;
}

long hx711_read_average(int samples = 5) {
  long sum = 0;
  int valid = 0;
  for (int i = 0; i < samples; i++) {
    if (hx711_is_ready()) {
      sum += hx711_read_raw();
      valid++;
    }
    delay(2);
  }
  return valid > 0 ? (sum / valid) : zero_offset;
}

void hx711_tare() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("{\"board\":3,\"status\":\"taring\",\"msg\":\"Zeroing load cell scale... leave empty.\"}");
  zero_offset = hx711_read_average(15);
  digitalWrite(LED_PIN, LOW);
}

// =============================================================================
// MPU-6050 Driver (Pure Wire.h)
// =============================================================================

bool writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(data);
  return (Wire.endTransmission() == 0);
}

bool readRegisters(uint8_t startReg, uint8_t* buffer, size_t count) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;

  size_t bytesRead = Wire.requestFrom((int)MPU_ADDR, (int)count, (int)true);
  if (bytesRead != count) return false;

  for (size_t i = 0; i < count; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

bool initMPU6050() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ);
  delay(50);

  uint8_t whoAmI = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoAmI, 1) || whoAmI != 0x68) {
    return false;
  }

  writeRegister(REG_PWR_MGMT_1, 0x01); // Clock source auto PLL
  delay(10);
  writeRegister(REG_SMPLRT_DIV, 0x00);
  writeRegister(REG_CONFIG, 0x03);       // 42 Hz low pass filter
  writeRegister(REG_GYRO_CONFIG, 0x00);  // +/- 250 deg/s
  writeRegister(REG_ACCEL_CONFIG, 0x00); // +/- 2g

  return true;
}

void calibrateGyro() {
  float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  const int SAMPLES = 100;
  for (int i = 0; i < SAMPLES; i++) {
    uint8_t buf[6];
    if (readRegisters(REG_GYRO_XOUT_H, buf, 6)) {
      int16_t raw_gx = (buf[0] << 8) | buf[1];
      int16_t raw_gy = (buf[2] << 8) | buf[3];
      int16_t raw_gz = (buf[4] << 8) | buf[5];
      gx_sum += (float)raw_gx / 131.0f;
      gy_sum += (float)raw_gy / 131.0f;
      gz_sum += (float)raw_gz / 131.0f;
    }
    delay(4);
  }
  gyro_x_offset = gx_sum / SAMPLES;
  gyro_y_offset = gy_sum / SAMPLES;
  gyro_z_offset = gz_sum / SAMPLES;
}

// =============================================================================
// Setup & Main Loop
// =============================================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Initialize HX711 Pins
  pinMode(HX711_DOUT, INPUT);
  pinMode(HX711_SCK, OUTPUT);
  digitalWrite(HX711_SCK, LOW);

  Serial.println("\n==================================================");
  Serial.println("ESP32 WEIGH STATION & KINEMATICS NODE (COM21)");
  Serial.println("Hardware: ESP32 + HX711 Load Cell + MPU-6050");
  Serial.println("I2C Pins: SDA = GPIO 21 | SCL = GPIO 22");
  Serial.println("HX711 Pins: DOUT = GPIO 18 | SCK = GPIO 19");
  Serial.println("Baud Rate: 115200");
  Serial.println("==================================================");

  // Initialize MPU-6050
  mpuConnected = initMPU6050();
  if (mpuConnected) {
    calibrateGyro();
    Serial.println("{\"board\":3,\"node\":\"Scale_IMU\",\"mpu\":\"ready\"}");
  } else {
    Serial.println("{\"board\":3,\"node\":\"Scale_IMU\",\"mpu\":\"waiting\"}");
  }

  // Quick check on HX711
  delay(100);
  if (hx711_is_ready()) {
    hx711_detected = true;
    hx711_tare();
    Serial.println("{\"board\":3,\"node\":\"Scale_IMU\",\"hx711\":\"ready\"}");
  } else {
    Serial.println("{\"board\":3,\"node\":\"Scale_IMU\",\"hx711\":\"waiting\"}");
  }

  prevTime = micros();
  lastTelemetryTime = millis();
  lastWeightReadTime = millis();
}

void loop() {
  // Listen for commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("TARE") || cmd.equalsIgnoreCase("ZERO")) {
      hx711_tare();
      pitch = 0.0f;
      roll  = 0.0f;
      yaw   = 0.0f;
      calibrateGyro();
      Serial.println("{\"board\":3,\"status\":\"tared\",\"msg\":\"Scale zeroed and orientation reset.\"}");
    } else if (cmd.startsWith("CAL:")) {
      float newCal = cmd.substring(4).toFloat();
      if (newCal > 1.0f) {
        scale_factor = newCal;
        Serial.printf("{\"board\":3,\"status\":\"cal_updated\",\"factor\":%.1f}\n", scale_factor);
      }
    }
  }

  // Read Weight at ~10 Hz (HX711 standard conversion rate)
  unsigned long nowMs = millis();
  if (nowMs - lastWeightReadTime >= 100) {
    lastWeightReadTime = nowMs;
    if (hx711_is_ready()) {
      long raw = hx711_read_raw();
      float w = (float)(raw - zero_offset) / scale_factor;
      if (abs(w) < 0.3f) w = 0.0f; // Noise gate around zero
      current_weight_g = w;
    }
  }

  // Update MPU-6050 Orientation
  unsigned long currentTimeMicros = micros();
  float dt = (currentTimeMicros - prevTime) / 1000000.0f;
  prevTime = currentTimeMicros;
  if (dt <= 0.0f || dt > 0.5f) dt = 0.02f;

  uint8_t buffer[14];
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0, tempC = 25.0f;

  if (mpuConnected && readRegisters(REG_ACCEL_XOUT_H, buffer, 14)) {
    int16_t raw_ax   = (buffer[0] << 8) | buffer[1];
    int16_t raw_ay   = (buffer[2] << 8) | buffer[3];
    int16_t raw_az   = (buffer[4] << 8) | buffer[5];
    int16_t raw_temp = (buffer[6] << 8) | buffer[7];
    int16_t raw_gx   = (buffer[8] << 8) | buffer[9];
    int16_t raw_gy   = (buffer[10] << 8) | buffer[11];
    int16_t raw_gz   = (buffer[12] << 8) | buffer[13];

    ax = (float)raw_ax / 16384.0f;
    ay = (float)raw_ay / 16384.0f;
    az = (float)raw_az / 16384.0f;
    tempC = ((float)raw_temp / 340.0f) + 36.53f;

    gx = ((float)raw_gx / 131.0f) - gyro_x_offset;
    gy = ((float)raw_gy / 131.0f) - gyro_y_offset;
    gz = ((float)raw_gz / 131.0f) - gyro_z_offset;

    float accel_roll  = atan2(ay, az) * 180.0f / PI;
    float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0f / PI;

    roll  = ALPHA * (roll  + gx * dt) + (1.0f - ALPHA) * accel_roll;
    pitch = ALPHA * (pitch + gy * dt) + (1.0f - ALPHA) * accel_pitch;
    yaw   += gz * dt;
  } else {
    // Retry connecting MPU-6050
    static unsigned long lastMpuRetry = 0;
    if (nowMs - lastMpuRetry > 1500) {
      lastMpuRetry = nowMs;
      mpuConnected = initMPU6050();
    }
  }

  // 50 Hz Telemetry Stream
  if (nowMs - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = nowMs;
    // Format: Scale_IMU -> Weight: 145.2 g | Accel: (0.012, -0.034, 0.985) g | Gyro: (0.12, -0.05, 0.01) dps | Pitch: 2.10 | Roll: -0.70 | Yaw: 0.15 | Temp: 28.5 C
    Serial.printf(
      "Scale_IMU -> Weight: %.1f g | Accel: (%.3f, %.3f, %.3f) g | Gyro: (%.2f, %.2f, %.2f) dps | Pitch: %.2f | Roll: %.2f | Yaw: %.2f | Temp: %.1f C\n",
      current_weight_g, ax, ay, az, gx, gy, gz, pitch, roll, yaw, tempC
    );
  }

  vTaskDelay(pdMS_TO_TICKS(1));
}
