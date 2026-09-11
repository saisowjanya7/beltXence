/*
  =============================================================================
  Dual ESP32 + MPU-6050 Telemetry Node
  Hardware:
    - ESP32 Dev Module (ESP32-D0WD-V3)
    - MPU-6050 6-Axis IMU (Accelerometer + Gyroscope + Temperature)
    - I2C Connection:
        * SDA  -> GPIO 21
        * SCL  -> GPIO 22
        * VCC  -> 3.3V
        * GND  -> GND
  
  Features:
    - Zero external library dependencies (uses native Wire.h)
    - 400kHz Fast I2C Bus Speed
    - Auto-calibration on boot (measures gyro baseline offsets)
    - Complementary Filter fusing Accel + Gyro for drift-free Pitch & Roll
    - High-rate JSON telemetry streamed over Serial (115200 baud) at 50Hz
    - Compatible with Web Serial API, Chrome/Edge Dashboard, and Arduino IDE
  =============================================================================
*/

#include <Wire.h>
#include <Arduino.h>

// ---------------------------------------------------------------------------
// BOARD CONFIGURATION
// Change to 1 for Board 1 (e.g. COM19) or 2 for Board 2 (e.g. COM21)
// ---------------------------------------------------------------------------
#define BOARD_ID 1

// Pin definitions as requested
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define I2C_FREQ    400000 // 400kHz Fast Mode

// MPU-6050 I2C Address (default: 0x68; if AD0 is HIGH: 0x69)
#define MPU6050_ADDR 0x68

// MPU-6050 Registers
#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_TEMP_OUT_H   0x41
#define REG_GYRO_XOUT_H  0x43
#define REG_PWR_MGMT_1   0x6B
#define REG_WHO_AM_I     0x75

// Telemetry output interval (50 Hz = 20 ms)
const unsigned long TELEMETRY_INTERVAL_MS = 20;

// Calibration offsets
float gyro_x_offset = 0.0f;
float gyro_y_offset = 0.0f;
float gyro_z_offset = 0.0f;

// Filtered orientation angles (degrees)
float pitch = 0.0f;
float roll = 0.0f;
float yaw = 0.0f;

// Complementary filter weight (alpha = gyro weight, 1-alpha = accel weight)
const float ALPHA = 0.98f;

unsigned long prevTime = 0;
unsigned long lastTelemetryTime = 0;
bool sensorConnected = false;

// ---------------------------------------------------------------------------
// Low-Level I2C Helper Functions
// ---------------------------------------------------------------------------
bool writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(data);
  return (Wire.endTransmission() == 0);
}

bool readRegisters(uint8_t startReg, uint8_t *buffer, uint8_t count) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  uint8_t received = Wire.requestFrom((uint8_t)MPU6050_ADDR, count);
  if (received != count) {
    return false;
  }
  for (uint8_t i = 0; i < count; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

// ---------------------------------------------------------------------------
// MPU-6050 Initialization
// ---------------------------------------------------------------------------
bool initMPU6050() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ);
  delay(100);

  // Check WHO_AM_I register (should return 0x68)
  uint8_t whoami = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoami, 1) || whoami != 0x68) {
    // Try alternate address if 0x68 fails
    Serial.printf("{\"status\":\"error\",\"board\":%d,\"msg\":\"MPU6050 not found at 0x68 (WHO_AM_I=0x%02X)\"}\n", BOARD_ID, whoami);
    return false;
  }

  // Wake up MPU6050 (clear sleep bit in PWR_MGMT_1)
  writeRegister(REG_PWR_MGMT_1, 0x01); // Clock Source = Auto Select (PLL with X Gyro)
  delay(10);

  // Sample Rate Divider = 0 (Sample Rate = 1kHz)
  writeRegister(REG_SMPLRT_DIV, 0x00);

  // DLPF Config: Bandwidth ~44Hz, delay 4.9ms
  writeRegister(REG_CONFIG, 0x03);

  // Gyro Config: Full Scale Range = ±250 °/s (FS_SEL = 0) -> Scale: 131.0 LSB/(°/s)
  writeRegister(REG_GYRO_CONFIG, 0x00);

  // Accel Config: Full Scale Range = ±2g (AFS_SEL = 0) -> Scale: 16384.0 LSB/g
  writeRegister(REG_ACCEL_CONFIG, 0x00);

  return true;
}

// ---------------------------------------------------------------------------
// Gyro Calibration
// ---------------------------------------------------------------------------
void calibrateGyro() {
  const int SAMPLES = 200;
  long gx_sum = 0, gy_sum = 0, gz_sum = 0;
  uint8_t rawData[6];

  for (int i = 0; i < SAMPLES; i++) {
    if (readRegisters(REG_GYRO_XOUT_H, rawData, 6)) {
      gx_sum += (int16_t)((rawData[0] << 8) | rawData[1]);
      gy_sum += (int16_t)((rawData[2] << 8) | rawData[3]);
      gz_sum += (int16_t)((rawData[4] << 8) | rawData[5]);
    }
    delay(3);
  }

  gyro_x_offset = (float)gx_sum / SAMPLES / 131.0f;
  gyro_y_offset = (float)gy_sum / SAMPLES / 131.0f;
  gyro_z_offset = (float)gz_sum / SAMPLES / 131.0f;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  sensorConnected = initMPU6050();
  if (sensorConnected) {
    calibrateGyro();
    Serial.printf("{\"status\":\"ready\",\"board\":%d,\"msg\":\"MPU6050 initialized on SDA:%d SCL:%d\"}\n",
                  BOARD_ID, I2C_SDA_PIN, I2C_SCL_PIN);
  } else {
    Serial.printf("{\"status\":\"warning\",\"board\":%d,\"msg\":\"Failed to init MPU6050. Check wiring SDA:%d SCL:%d\"}\n",
                  BOARD_ID, I2C_SDA_PIN, I2C_SCL_PIN);
  }

  prevTime = micros();
  lastTelemetryTime = millis();
}

// ---------------------------------------------------------------------------
// Main Loop
// ---------------------------------------------------------------------------
void loop() {
  // Check for incoming serial commands (e.g., "TARE" or "CAL")
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("TARE") || cmd.equalsIgnoreCase("CAL")) {
      pitch = 0.0f;
      roll = 0.0f;
      yaw = 0.0f;
      calibrateGyro();
      Serial.printf("{\"status\":\"calibrated\",\"board\":%d}\n", BOARD_ID);
    }
  }

  unsigned long currentTimeMicros = micros();
  float dt = (currentTimeMicros - prevTime) / 1000000.0f;
  prevTime = currentTimeMicros;

  // Protect against micros() overflow or huge pauses
  if (dt <= 0.0f || dt > 0.5f) {
    dt = 0.02f;
  }

  // 14 Bytes burst read: Accel (6) + Temp (2) + Gyro (6)
  uint8_t buffer[14];
  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  float tempC = 0;

  if (sensorConnected && readRegisters(REG_ACCEL_XOUT_H, buffer, 14)) {
    int16_t raw_ax = (buffer[0] << 8) | buffer[1];
    int16_t raw_ay = (buffer[2] << 8) | buffer[3];
    int16_t raw_az = (buffer[4] << 8) | buffer[5];
    int16_t raw_temp = (buffer[6] << 8) | buffer[7];
    int16_t raw_gx = (buffer[8] << 8) | buffer[9];
    int16_t raw_gy = (buffer[10] << 8) | buffer[11];
    int16_t raw_gz = (buffer[12] << 8) | buffer[13];

    // Convert raw values to physical engineering units:
    // Accel: ±2g range -> 16384 LSB/g
    ax = (float)raw_ax / 16384.0f;
    ay = (float)raw_ay / 16384.0f;
    az = (float)raw_az / 16384.0f;

    // Temperature in °C: Temp_degC = (raw_temp / 340.0) + 36.53
    tempC = ((float)raw_temp / 340.0f) + 36.53f;

    // Gyro: ±250 °/s range -> 131.0 LSB/(°/s), minus offset
    gx = ((float)raw_gx / 131.0f) - gyro_x_offset;
    gy = ((float)raw_gy / 131.0f) - gyro_y_offset;
    gz = ((float)raw_gz / 131.0f) - gyro_z_offset;

    // Accelerometer-based angle calculation (degrees)
    float accel_roll  = atan2(ay, az) * 180.0f / PI;
    float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0f / PI;

    // Complementary Filter Fusion
    roll  = ALPHA * (roll  + gx * dt) + (1.0f - ALPHA) * accel_roll;
    pitch = ALPHA * (pitch + gy * dt) + (1.0f - ALPHA) * accel_pitch;
    yaw   += gz * dt; // Gyro integration for relative yaw
  } else {
    // Retry sensor connection every 1 second if disconnected
    static unsigned long lastRetry = 0;
    if (millis() - lastRetry > 1000) {
      lastRetry = millis();
      sensorConnected = initMPU6050();
    }
  }

  // High-Rate Telemetry Broadcast
  unsigned long now = millis();
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;

    // Compact JSON packet format
    Serial.printf(
      "{\"board\":%d,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"pitch\":%.2f,\"roll\":%.2f,\"yaw\":%.2f,\"temp\":%.1f,\"ms\":%lu}\n",
      BOARD_ID, ax, ay, az, gx, gy, gz, pitch, roll, yaw, tempC, now
    );
  }

  // Yield to ESP32 RTOS background tasks
  vTaskDelay(pdMS_TO_TICKS(1));
}
