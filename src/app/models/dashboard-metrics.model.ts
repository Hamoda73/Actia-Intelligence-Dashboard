// src/app/models/dashboard-metrics.model.ts
export interface DashboardMetrics {
  // Core battery parameters from your BatteryData model
  batteryVoltage: number;
  motorCurrent: number;
  stateOfCharge: number;
  maxCellVoltage: number;
  minCellVoltage: number;
  maxTemperature: number;
  minTemperature: number;
  vehicleState: string;
  chargeState: string;
  
  // Device information
  deviceId: string;
  sensorId: string;
  type: string;
  timestamp: string;
  
  // Display timestamp
  lastUpdated: string;
}