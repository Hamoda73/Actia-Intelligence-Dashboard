// src/app/models/prediction.model.ts (note: fixed filename typo)
export interface PredictionRisk {
  level: 'low' | 'medium' | 'high';
  percentage: number;
  description: string;
  recommendation: string;
}

export interface SystemPredictions {
  // EV Battery specific predictions
  batteryRisk: PredictionRisk;
  voltageRisk: PredictionRisk;
  cellDegradationRisk: PredictionRisk;
  
  // Battery health metrics
  overallHealth: number;
  temperatureEfficiency: number;
  chargeCycles: number;
  
  lastUpdated: string;
}