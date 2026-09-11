/*
  =============================================================================
  Dual ESP32 + MPU-6050 Telemetry Node - Robust Edition
  Board 2 (COM21)
  Wiring:
    * MPU-6050 VCC -> ESP32 3.3V (or 5V for modules with 3.3V LDO)
    * MPU-6050 GND -> ESP32 GND
    * MPU-6050 SDA -> ESP32 GPIO 21
    * MPU-6050 SCL -> ESP32 GPIO 22
    * MPU-6050 AD0 -> GND (Address 0x68) or 3.3V (Address 0x69)
  =============================================================================
*/

#include <Wire.h>
#include <Arduino.h>

#define BOARD_ID 2

#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define I2C_FREQ    400000

#define LED_PIN     2 // Onboard Blue LED on most ESP32 DevKit boards

// Supported MPU6050 addresses (0x68 when AD0=GND, 0x69 when AD0=VCC)
uint8_t mpu_addr = 0x68;

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_TEMP_OUT_H   0x41
#define REG_GYRO_XOUT_H  0x43
#define REG_PWR_MGMT_1   0x6B
#define REG_WHO_AM_I     0x75

const unsigned long TELEMETRY_INTERVAL_MS = 20; // 50 Hz

float gyro_x_offset = 0.0f;
float gyro_y_offset = 0.0f;
float gyro_z_offset = 0.0f;

float pitch = 0.0f;
float roll = 0.0f;
float yaw = 0.0f;
const float ALPHA = 0.98f;

unsigned long prevTime = 0;
unsigned long lastTelemetryTime = 0;
bool sensorConnected = false;

// ---------------------------------------------------------------------------
// Low-Level I2C Helper Functions
// ---------------------------------------------------------------------------
bool writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(mpu_addr);
  Wire.write(reg);
  Wire.write(data);
  return (Wire.endTransmission() == 0);
}

bool readRegisters(uint8_t startReg, uint8_t *buffer, uint8_t count) {
  Wire.beginTransmission(mpu_addr);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;
  uint8_t received = Wire.requestFrom((uint8_t)mpu_addr, count);
  if (received != count) return false;
  for (uint8_t i = 0; i < count; i++) {
    buffer[i] = Wire.read();
  }
  return true;
}

// ---------------------------------------------------------------------------
// I2C Bus Scanner & Auto-Detection
// ---------------------------------------------------------------------------
bool detectMPU6050() {
  uint8_t addresses[] = {0x68, 0x69};
  for (uint8_t addr : addresses) {
    Wire.beginTransmission(addr);
    Wire.write(REG_WHO_AM_I);
    if (Wire.endTransmission(false) == 0) {
      if (Wire.requestFrom((uint8_t)addr, (uint8_t)1) == 1) {
        uint8_t whoami = Wire.read();
        if (whoami == 0x68 || whoami == 0x70 || whoami == 0x71 || whoami == 0x72) {
          mpu_addr = addr;
          return true;
        }
      }
    }
  }
  return false;
}

bool initMPU6050() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ);
  delay(100);

  if (!detectMPU6050()) {
    return false;
  }

  // Wake up MPU6050 (clear sleep bit in PWR_MGMT_1)
  writeRegister(REG_PWR_MGMT_1, 0x01);
  delay(15);
  writeRegister(REG_SMPLRT_DIV, 0x00);
  writeRegister(REG_CONFIG, 0x03);       // DLPF ~44Hz
  writeRegister(REG_GYRO_CONFIG, 0x00);  // ±250 °/s
  writeRegister(REG_ACCEL_CONFIG, 0x00); // ±2g
  return true;
}

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

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  delay(500);

  sensorConnected = initMPU6050();
  if (sensorConnected) {
    digitalWrite(LED_PIN, HIGH);
    calibrateGyro();
    Serial.printf("{\"board\":%d,\"status\":\"ready\",\"addr\":\"0x%02X\",\"sda\":%d,\"scl\":%d}\n",
                  BOARD_ID, mpu_addr, I2C_SDA_PIN, I2C_SCL_PIN);
  } else {
    Serial.printf("{\"board\":%d,\"status\":\"error\",\"msg\":\"MPU6050 not found on SDA:21 SCL:22. Check wiring VCC/GND/SDA/SCL\"}\n", BOARD_ID);
  }

  prevTime = micros();
  lastTelemetryTime = millis();
}

void loop() {
  // Listen for commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("TARE") || cmd.equalsIgnoreCase("CAL")) {
      pitch = 0.0f; roll = 0.0f; yaw = 0.0f;
      calibrateGyro();
      Serial.printf("{\"board\":%d,\"status\":\"calibrated\"}\n", BOARD_ID);
    }
  }

  unsigned long currentTimeMicros = micros();
  float dt = (currentTimeMicros - prevTime) / 1000000.0f;
  prevTime = currentTimeMicros;
  if (dt <= 0.0f || dt > 0.5f) dt = 0.02f;

  uint8_t buffer[14];
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0, tempC = 0;

  if (sensorConnected && readRegisters(REG_ACCEL_XOUT_H, buffer, 14)) {
    digitalWrite(LED_PIN, HIGH);
    int16_t raw_ax = (buffer[0] << 8) | buffer[1];
    int16_t raw_ay = (buffer[2] << 8) | buffer[3];
    int16_t raw_az = (buffer[4] << 8) | buffer[5];
    int16_t raw_temp = (buffer[6] << 8) | buffer[7];
    int16_t raw_gx = (buffer[8] << 8) | buffer[9];
    int16_t raw_gy = (buffer[10] << 8) | buffer[11];
    int16_t raw_gz = (buffer[12] << 8) | buffer[13];

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
    digitalWrite(LED_PIN, (millis() / 250) % 2); // Blink rapidly if sensor missing
    static unsigned long lastRetry = 0;
    if (millis() - lastRetry > 1000) {
      lastRetry = millis();
      sensorConnected = initMPU6050();
      if (!sensorConnected) {
        Serial.printf("{\"board\":%d,\"status\":\"error\",\"msg\":\"Waiting for MPU6050 on SDA:21 SCL:22...\"}\n", BOARD_ID);
      }
    }
  }

  unsigned long now = millis();
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;
    Serial.printf(
      "{\"board\":%d,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"pitch\":%.2f,\"roll\":%.2f,\"yaw\":%.2f,\"temp\":%.1f,\"ms\":%lu}\n",
      BOARD_ID, ax, ay, az, gx, gy, gz, pitch, roll, yaw, tempC, now
    );
  }
  vTaskDelay(pdMS_TO_TICKS(1));
}
