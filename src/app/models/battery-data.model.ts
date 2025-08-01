export interface BatteryData {
  id: string;
  evdata: string;
  deviceId: string;
  sensorId: string;
  type: string;
  vehicleState: string;
  chargeState: string;
  batteryVoltage: number;
  motorCurrent: number;
  stateOfCharge: number;
  maxCellVoltage: number;
  minCellVoltage: number;
  maxTemperature: number;
  minTemperature: number;
  timestamp: string;
  metadata?: Metadata;
}

export interface Metadata {
  source: string;
  row_index: number;
}