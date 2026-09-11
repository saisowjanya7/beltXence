// ============================================================
// Arduino Uno - E18-D80NK Digital IR Obstacle / Proximity Sensor
// Hardware:
//   - Arduino Uno
//   - E18-D80NK Infrared Obstacle Sensor (or digital IR module)
//   - Wiring:
//       * Brown Wire  -> 5V
//       * Blue Wire   -> GND
//       * Black Wire  -> Pin 2 (Signal with internal pull-up)
//       * Built-in LED -> Pin 13
// ============================================================

const int irSensorPin  = 2;           // Signal wire (Black)
const int statusLedPin = LED_BUILTIN; // Onboard LED (Pin 13)

void setup() {
  Serial.begin(9600); // 9600 baud for Arduino Uno
  
  // Enable internal pull-up resistor (crucial for NPN open-collector outputs)
  pinMode(irSensorPin, INPUT_PULLUP);
  pinMode(statusLedPin, OUTPUT);

  Serial.println("E18-D80NK IR Sensor Initialized.");
}

void loop() {
  int sensorState = digitalRead(irSensorPin);

  // NPN sensor pulls the output LOW when an obstacle is detected
  if (sensorState == LOW) {
    digitalWrite(statusLedPin, HIGH); // Turn ON built-in LED
    Serial.println("Obstacle Detected!");
  } else {
    digitalWrite(statusLedPin, LOW);  // Turn OFF built-in LED
    Serial.println("Clear");
  }

  delay(150); // Small debounce/readability delay
}
