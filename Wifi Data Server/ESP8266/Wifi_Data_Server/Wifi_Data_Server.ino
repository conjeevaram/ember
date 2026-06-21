void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("ESP8266 alive");
}

void loop() {
  digitalWrite(LED_BUILTIN, LOW);   // on (active low)
  delay(500);
  digitalWrite(LED_BUILTIN, HIGH);  // off
  delay(500);
  Serial.println("blink");
}
