// =============================================================================
// Arduino Uno - Conveyor Belt Location Tracker & Real-Time Speedometer
// Hardware:
//   - Arduino Uno
//   - E18-D80NK Digital Optical / IR Sensor (or TCRT5000 / Line Sensor)
//   - Wiring:
//       * Brown Wire  -> 5V
//       * Blue Wire   -> GND
//       * Black Wire  -> Pin 2 (with INPUT_PULLUP)
//       * Onboard LED -> Pin 13
//
// Measurement Principle:
//   - Beam is HIGH when clear, pulls LOW when white line or object enters beam.
//   - Microsecond transit timer measures exact passage duration:
//       dt_micros = Exit_Time - Enter_Time
//       Real-Time Velocity (cm/s) = (Object_Length_cm * 1,000,000) / dt_micros
//   - Fast movement  -> short dt -> HIGH SPEED
//   - Slow movement  -> long dt  -> LOW SPEED
//   - Immediately after detecting, executes a 4-SECOND POST-DETECTION DELAY
//     with live countdown updates, then advances to next location (Loc 1->2->3->4).
// =============================================================================

const int irSensorPin  = 2;           // IR sensor signal pin (INT0)
const int statusLedPin = LED_BUILTIN; // Pin 13 LED indicator

// Sensor & Object Physical Dimensions
// Adjust OBJECT_LENGTH_CM to match your marker width or package length
const float OBJECT_LENGTH_CM = 8.0;   // Length of passing object/marker in cm
const int TOTAL_LOCATIONS    = 4;     // Total numbered station locations

// State variables
int currentLocation          = 1;     // Active station location (1..4)
float currentSpeed_cm_s      = 0.0;   // Real-time measured velocity
int lastPinState             = HIGH;  // Previous polled pin state

void setup() {
  Serial.begin(9600); // 9600 baud for Arduino Uno

  pinMode(irSensorPin, INPUT_PULLUP);
  pinMode(statusLedPin, OUTPUT);
  digitalWrite(statusLedPin, LOW);

  lastPinState = digitalRead(irSensorPin);

  Serial.println("==================================================");
  Serial.println("CONVEYOR BELT TRACKER & REAL-TIME SPEEDOMETER");
  Serial.print("Target Object / Marker Length: ");
  Serial.print(OBJECT_LENGTH_CM, 1);
  Serial.println(" cm");
  Serial.println("Post-Detection Delay: 4 SECONDS ACTIVE");
  Serial.println("Telemetry: Active @ 9600 Baud");
  Serial.println("==================================================");

  // Initial state announcement
  Serial.print("Belt -> Loc: ");
  Serial.print(currentLocation);
  Serial.println(" | Speed: 0.0 cm/s | State: Ready / Infeed Station");
}

void loop() {
  // Read current IR sensor state (Active-LOW: LOW = Object/Line detected, HIGH = Clear)
  int currentPin = digitalRead(irSensorPin);

  // Detect Leading Edge: Beam is broken (HIGH -> LOW)
  if (lastPinState == HIGH && currentPin == LOW) {
    unsigned long tEnter = micros();
    digitalWrite(statusLedPin, HIGH);

    // Measure exact microsecond duration while beam is broken
    // Timeout of 2.0 seconds prevents lockup if object rests permanently
    while (digitalRead(irSensorPin) == LOW && (micros() - tEnter < 2000000)) {
      delayMicroseconds(100);
    }
    unsigned long tExit = micros();
    unsigned long transitMicros = tExit - tEnter;

    // Debounce: ignore noise glitches shorter than 4ms
    if (transitMicros >= 4000) {
      // 1. Calculate Real-Time Velocity from actual passage duration!
      // Speed (cm/s) = (Length in cm * 1,000,000) / Duration in microseconds
      float measuredSpeed_cm_s = (OBJECT_LENGTH_CM * 1000000.0) / (float)transitMicros;

      // Cap to reasonable physical limits (1 cm/s to 300 cm/s)
      if (measuredSpeed_cm_s < 1.0) measuredSpeed_cm_s = 1.0;
      if (measuredSpeed_cm_s > 300.0) measuredSpeed_cm_s = 300.0;

      currentSpeed_cm_s = measuredSpeed_cm_s;
      float speed_m_s = currentSpeed_cm_s / 100.0;
      unsigned long transit_ms = transitMicros / 1000;

      // 2. Output instant detection telemetry with measured velocity
      Serial.print("Belt -> Loc: ");
      Serial.print(currentLocation);
      Serial.print(" | Speed: ");
      Serial.print(currentSpeed_cm_s, 1);
      Serial.print(" cm/s (");
      Serial.print(speed_m_s, 2);
      Serial.print(" m/s) | Transit: ");
      Serial.print(transit_ms);
      Serial.print(" ms | Marker: Location ");
      Serial.print(currentLocation);
      Serial.println(" -> DETECTED!");

      // 3. Execute 4-Second Delay after detecting (with real-time countdown)
      for (int s = 4; s >= 1; s--) {
        Serial.print("Belt -> Loc: ");
        Serial.print(currentLocation);
        Serial.print(" | Speed: ");
        Serial.print(currentSpeed_cm_s, 1);
        Serial.print(" cm/s | State: Station ");
        Serial.print(currentLocation);
        Serial.print(" Holding (");
        Serial.print(s);
        Serial.println("s remaining)");

        // Flash LED to visibly show 4-second hold
        digitalWrite(statusLedPin, (s % 2 == 0) ? HIGH : LOW);
        delay(1000);
      }

      digitalWrite(statusLedPin, LOW);

      // 4. Advance to next station location
      currentLocation++;
      if (currentLocation > TOTAL_LOCATIONS) {
        currentLocation = 1;
      }

      Serial.print("Belt -> Loc: ");
      Serial.print(currentLocation);
      Serial.print(" | Speed: ");
      Serial.print(currentSpeed_cm_s, 1);
      Serial.print(" cm/s | State: 4s Delay Complete -> Heading to Location ");
      Serial.println(currentLocation);

      // Debounce delay so sensor clears before next read
      delay(50);
      currentPin = digitalRead(irSensorPin);
    }
  }

  lastPinState = currentPin;

  // Periodic heartbeat every 1500 ms while waiting for next object/marker
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat >= 1500) {
    lastHeartbeat = millis();

    Serial.print("Belt -> Loc: ");
    Serial.print(currentLocation);
    Serial.print(" | Speed: ");
    Serial.print(currentSpeed_cm_s, 1);
    Serial.print(" cm/s | State: Ready -> Waiting for Location ");
    Serial.print(currentLocation);
    Serial.print(" | Pin2: ");
    Serial.println(digitalRead(irSensorPin) == LOW ? "LOW (Object in Beam)" : "HIGH (Clear)");
  }

  delay(10);
}
