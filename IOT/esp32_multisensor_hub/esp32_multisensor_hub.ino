#include <Wire.h>

// ============================================================
// Device Addresses & Pin Definitions
// ============================================================
const int MPU_ADDR    = 0x68; // MPU-6050 I2C address (SDA=21, SCL=22)
const int MLX_ADDR    = 0x5A; // MLX90614 I2C address (SDA=21, SCL=22)
const int HX711_DOUT  = 18;   // Load cell Data pin (GPIO 18)
const int HX711_SCK   = 19;   // Load cell Clock pin (GPIO 19)

// Load cell calibration factor and baseline offset
float scale_factor = 2280.0;
long zero_offset   = 0;

// Variables to store sensor readings
float ax, ay, az;
float gx, gy, gz;
float mpu_temp;
float mlx_ambient, mlx_object;

// ============================================================
// MLX90614 Driver Functions (Pure Wire.h)
// ============================================================

// Reads a 16-bit temperature register (0x06 = Ambient, 0x07 = Object)
float mlx90614_read_temp(uint8_t reg) {
  Wire.beginTransmission(MLX_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false); // Repeated start condition required by SMBus

  // Request 3 bytes: Data LSB, Data MSB, and PEC (Packet Error Code)
  Wire.requestFrom(MLX_ADDR, 3, true);
  if (Wire.available() >= 3) {
    uint8_t lsb = Wire.read();
    uint8_t msb = Wire.read();
    uint8_t pec = Wire.read(); // Discard PEC byte

    uint16_t rawData = (msb << 8) | lsb;

    // Convert raw reading to Celsius:
    // Resolution is 0.02°K per LSB. 0x0000 = -273.15°C
    float tempKelvin = (float)rawData * 0.02;
    return tempKelvin - 273.15;
  }
  return -999.0; // Return error value if communication fails
}

// ============================================================
// MPU-6050 Driver Functions (Pure Wire.h)
// ============================================================

void mpu6050_init() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Clear sleep mode bit
  Wire.endTransmission(true);
  Serial.println("MPU-6050: Initialized.");
}

void mpu6050_read() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Starting register: ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  if (Wire.available() >= 14) {
    int16_t rawAcX = Wire.read() << 8 | Wire.read();
    int16_t rawAcY = Wire.read() << 8 | Wire.read();
    int16_t rawAcZ = Wire.read() << 8 | Wire.read();
    int16_t rawTmp = Wire.read() << 8 | Wire.read();
    int16_t rawGyX = Wire.read() << 8 | Wire.read();
    int16_t rawGyY = Wire.read() << 8 | Wire.read();
    int16_t rawGyZ = Wire.read() << 8 | Wire.read();

    ax = (float)rawAcX / 16384.0;
    ay = (float)rawAcY / 16384.0;
    az = (float)rawAcZ / 16384.0;
    gx = (float)rawGyX / 131.0;
    gy = (float)rawGyY / 131.0;
    gz = (float)rawGyZ / 131.0;
    mpu_temp = (rawTmp / 340.0) + 36.53;
  }
}

// ============================================================
// HX711 Load Cell Driver Functions (Bit-Banging)
// ============================================================

bool hx711_is_ready() {
  return digitalRead(HX711_DOUT) == LOW;
}

long hx711_read() {
  unsigned long timeout = millis();
  while (!hx711_is_ready()) {
    if (millis() - timeout > 200) return 0;
    delay(1);
  }

  unsigned long count = 0;
  for (int i = 0; i < 24; i++) {
    digitalWrite(HX711_SCK, HIGH);
    delayMicroseconds(1);
    count = (count << 1);
    if (digitalRead(HX711_DOUT)) count++;
    digitalWrite(HX711_SCK, LOW);
    delayMicroseconds(1);
  }

  // 25th clock pulse (Channel A, Gain 128)
  digitalWrite(HX711_SCK, HIGH);
  delayMicroseconds(1);
  digitalWrite(HX711_SCK, LOW);
  delayMicroseconds(1);

  if (count & 0x800000) count |= 0xFF000000;
  return (long)count;
}

long hx711_read_average(byte samples) {
  long sum = 0;
  for (byte i = 0; i < samples; i++) {
    sum += hx711_read();
    delay(2);
  }
  return sum / samples;
}

void hx711_tare(byte samples) {
  zero_offset = hx711_read_average(samples);
}

float hx711_get_units(byte samples) {
  long raw = hx711_read_average(samples);
  return (float)(raw - zero_offset) / scale_factor;
}

// ============================================================
// Setup & Loop
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(100);

  // 1. Initialize shared I2C bus (SDA: GPIO 21, SCL: GPIO 22)
  Wire.begin(21, 22);

  // 2. Initialize MPU-6050
  mpu6050_init();

  // 3. Initialize HX711 pins and tare
  pinMode(HX711_DOUT, INPUT);
  pinMode(HX711_SCK, OUTPUT);
  digitalWrite(HX711_SCK, LOW);

  Serial.println("HX711: Zeroing scale... leave empty.");
  hx711_tare(15);
  Serial.println("System ready.");
}

void loop() {
  // 1. Read MPU-6050
  mpu6050_read();

  // 2. Read MLX90614 (0x06 = Ambient, 0x07 = Object Target)
  mlx_ambient = mlx90614_read_temp(0x06);
  mlx_object  = mlx90614_read_temp(0x07);

  // 3. Read Load Cell
  float weight = 0.0;
  if (hx711_is_ready()) {
    weight = hx711_get_units(2);
  }

  // 4. Print all values
  Serial.print("Weight: ");
  Serial.print(weight, 1);
  Serial.print(" g");

  Serial.print(" | IR Obj: ");
  Serial.print(mlx_object, 1);
  Serial.print(" C | IR Amb: ");
  Serial.print(mlx_ambient, 1);
  Serial.print(" C");

  Serial.print(" | Accel [g]: (");
  Serial.print(ax, 2); Serial.print(", ");
  Serial.print(ay, 2); Serial.print(", ");
  Serial.print(az, 2); Serial.print(")");

  Serial.print(" | Gyro [dps]: (");
  Serial.print(gx, 1); Serial.print(", ");
  Serial.print(gy, 1); Serial.print(", ");
  Serial.print(gz, 1); Serial.println(")");

  delay(60);
}
