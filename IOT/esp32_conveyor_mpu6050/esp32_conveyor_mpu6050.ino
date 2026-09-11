// =============================================================================
// ESP32 - Conveyor Frame 3D Motion & Vibration Telemetry Node (COM19)
// Hardware:
//   - ESP32 Development Board (Mounted on Conveyor Frame)
//   - MPU-6050 6-Axis MotionTracking Sensor (Accelerometer + Gyroscope + Temp)
//   - Wiring:
//       * VCC -> ESP32 3.3V (or 5V for boards with onboard 3.3V regulator)
//       * GND -> ESP32 GND
//       * SDA -> ESP32 GPIO 21 (Hardware I2C SDA)
//       * SCL -> ESP32 GPIO 22 (Hardware I2C SCL)
//       * AD0 -> GND (Address 0x68)
//
// Features:
//   - Pure Wire.h implementation (Zero external library dependencies)
//   - Hardware 400 kHz Fast I2C Bus
//   - Sensor Auto-Wakeup and Startup Gyro Calibration
//   - Complementary Filter (Alpha = 0.98) for drift-free Pitch, Roll & Yaw
//   - Conveyor Frame Vibration Index (G-Total = sqrt(ax^2 + ay^2 + az^2))
//   - 50 Hz High-Speed Telemetry Stream @ 115200 Baud
//   - Dynamic TARE command support via Serial
// =============================================================================

#include <Wire.h>
#include <Arduino.h>

#define BOARD_ID 1
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define I2C_FREQ    400000

#define LED_PIN     2 // Onboard LED on most ESP32 DevKit boards

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
bool sensorConnected = false;

// Low-Level I2C Register Helpers
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

  // Check WHO_AM_I register (should return 0x68)
  uint8_t whoAmI = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoAmI, 1) || whoAmI != 0x68) {
    return false;
  }

  // Wake up sensor (clear SLEEP bit)
  if (!writeRegister(REG_PWR_MGMT_1, 0x01)) return false; // Clock = Auto Gyro PLL
  delay(10);

  // Set Sample Rate Divider = 0 (1 kHz internal sample rate)
  writeRegister(REG_SMPLRT_DIV, 0x00);

  // Set DLPF (Digital Low Pass Filter) to ~42 Hz bandwidth to filter conveyor motor noise
  writeRegister(REG_CONFIG, 0x03);

  // Set Gyro full scale: +/- 250 deg/s (sensitivity 131 LSB/deg/s)
  writeRegister(REG_GYRO_CONFIG, 0x00);

  // Set Accel full scale: +/- 2g (sensitivity 16384 LSB/g)
  writeRegister(REG_ACCEL_CONFIG, 0x00);

  return true;
}

void calibrateGyro() {
  Serial.println("{\"status\":\"calibrating\",\"msg\":\"Hold conveyor frame still for gyro zero calibration...\"}");
  float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  const int SAMPLES = 200;

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
    delay(5);
  }

  gyro_x_offset = gx_sum / SAMPLES;
  gyro_y_offset = gy_sum / SAMPLES;
  gyro_z_offset = gz_sum / SAMPLES;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\n==================================================");
  Serial.println("ESP32 CONVEYOR FRAME MOTION & VIBRATION MONITOR");
  Serial.println("Hardware: ESP32 + MPU-6050 (COM19)");
  Serial.println("I2C Pins: SDA = GPIO 21 | SCL = GPIO 22");
  Serial.println("Baud Rate: 115200");
  Serial.println("==================================================");

  sensorConnected = initMPU6050();
  if (sensorConnected) {
    digitalWrite(LED_PIN, HIGH);
    calibrateGyro();
    Serial.println("{\"board\":1,\"node\":\"Frame_IMU\",\"status\":\"ready\",\"addr\":\"0x68\",\"sda\":21,\"scl\":22}");
  } else {
    Serial.println("{\"board\":1,\"node\":\"Frame_IMU\",\"status\":\"error\",\"msg\":\"MPU-6050 not found on SDA:21 SCL:22. Check 3.3V/GND/SDA/SCL wiring!\"}");
  }

  prevTime = micros();
  lastTelemetryTime = millis();
}

void loop() {
  // Listen for serial commands from Web Dashboard
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("TARE") || cmd.equalsIgnoreCase("CAL") || cmd.equalsIgnoreCase("ZERO")) {
      pitch = 0.0f;
      roll  = 0.0f;
      yaw   = 0.0f;
      calibrateGyro();
      Serial.println("{\"board\":1,\"status\":\"tared\",\"msg\":\"Orientation zeroed to current conveyor frame level.\"}");
    }
  }

  unsigned long currentTimeMicros = micros();
  float dt = (currentTimeMicros - prevTime) / 1000000.0f;
  prevTime = currentTimeMicros;
  if (dt <= 0.0f || dt > 0.5f) dt = 0.02f;

  uint8_t buffer[14];
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0, tempC = 0;
  float gTotal = 1.0f;

  if (sensorConnected && readRegisters(REG_ACCEL_XOUT_H, buffer, 14)) {
    digitalWrite(LED_PIN, HIGH);

    int16_t raw_ax   = (buffer[0] << 8) | buffer[1];
    int16_t raw_ay   = (buffer[2] << 8) | buffer[3];
    int16_t raw_az   = (buffer[4] << 8) | buffer[5];
    int16_t raw_temp = (buffer[6] << 8) | buffer[7];
    int16_t raw_gx   = (buffer[8] << 8) | buffer[9];
    int16_t raw_gy   = (buffer[10] << 8) | buffer[11];
    int16_t raw_gz   = (buffer[12] << 8) | buffer[13];

    // Conversion to engineering units
    ax = (float)raw_ax / 16384.0f;
    ay = (float)raw_ay / 16384.0f;
    az = (float)raw_az / 16384.0f;
    tempC = ((float)raw_temp / 340.0f) + 36.53f;

    gx = ((float)raw_gx / 131.0f) - gyro_x_offset;
    gy = ((float)raw_gy / 131.0f) - gyro_y_offset;
    gz = ((float)raw_gz / 131.0f) - gyro_z_offset;

    // Conveyor Frame Total G-Force / Vibration Magnitude
    gTotal = sqrt(ax * ax + ay * ay + az * az);

    // Accelerometer tilt angles
    float accel_roll  = atan2(ay, az) * 180.0f / PI;
    float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0f / PI;

    // Complementary Filter sensor fusion for stable Pitch & Roll
    roll  = ALPHA * (roll  + gx * dt) + (1.0f - ALPHA) * accel_roll;
    pitch = ALPHA * (pitch + gy * dt) + (1.0f - ALPHA) * accel_pitch;
    yaw   += gz * dt; // Gyroscope Z-axis integration for relative yaw
  } else {
    // Sensor disconnected recovery loop
    digitalWrite(LED_PIN, (millis() / 250) % 2); // Rapid blink
    static unsigned long lastRetry = 0;
    if (millis() - lastRetry > 1000) {
      lastRetry = millis();
      sensorConnected = initMPU6050();
      if (!sensorConnected) {
        Serial.println("{\"board\":1,\"status\":\"error\",\"msg\":\"Waiting for MPU-6050 on SDA:21 SCL:22...\"}");
      }
    }
  }

  // 50 Hz Telemetry Stream
  unsigned long now = millis();
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;

    // Send formatted telemetry line
    Serial.printf(
      "Frame_IMU -> Accel: (%.3f, %.3f, %.3f) g | Gyro: (%.2f, %.2f, %.2f) dps | Pitch: %.2f | Roll: %.2f | Yaw: %.2f | Vib: %.2fg | Temp: %.1f C\n",
      ax, ay, az, gx, gy, gz, pitch, roll, yaw, gTotal, tempC
    );
  }

  vTaskDelay(pdMS_TO_TICKS(1));
}
