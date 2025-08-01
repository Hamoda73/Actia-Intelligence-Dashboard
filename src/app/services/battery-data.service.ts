// src/app/services/battery-data.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, interval } from 'rxjs';
import { map, catchError, switchMap } from 'rxjs/operators';
import { BatteryData } from '../models/battery-data.model';
import { DashboardMetrics } from '../models/dashboard-metrics.model'; 
import { SystemPredictions, PredictionRisk } from '../models/pediction.model'; 

@Injectable({
  providedIn: 'root'
})
export class BatteryDataService {
  private readonly API_BASE_URL = 'http://localhost:8080/api/battery';
  
  private metricsSubject = new BehaviorSubject<DashboardMetrics | null>(null);
  private predictionsSubject = new BehaviorSubject<SystemPredictions | null>(null);
  
  public metrics$ = this.metricsSubject.asObservable();
  public predictions$ = this.predictionsSubject.asObservable();

  constructor(private http: HttpClient) {
    this.startRealTimeUpdates();
  }

  private httpOptions = {
    headers: new HttpHeaders({
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    })
  };

  // Get all battery data
  getAllBatteryData(): Observable<BatteryData[]> {
    return this.http.get<BatteryData[]>(this.API_BASE_URL, this.httpOptions)
      .pipe(
        catchError(this.handleError<BatteryData[]>('getAllBatteryData', []))
      );
  }

  // Get latest battery data
  getLatestBatteryData(): Observable<BatteryData> {
    return this.http.get<BatteryData>(`${this.API_BASE_URL}/latest`, this.httpOptions)
      .pipe(
        catchError(this.handleError<BatteryData>('getLatestBatteryData'))
      );
  }

  // Transform battery data to dashboard metrics - now using ALL your actual data fields
  private transformToDashboardMetrics(data: BatteryData): DashboardMetrics {
    return {
      // Direct mapping from your BatteryData model
      batteryVoltage: data.batteryVoltage,
      motorCurrent: data.motorCurrent,
      stateOfCharge: data.stateOfCharge,
      maxCellVoltage: data.maxCellVoltage,
      minCellVoltage: data.minCellVoltage,
      maxTemperature: data.maxTemperature,
      minTemperature: data.minTemperature,
      vehicleState: data.vehicleState,
      chargeState: data.chargeState,
      deviceId: data.deviceId,
      sensorId: data.sensorId,
      type: data.type,
      timestamp: data.timestamp,
      lastUpdated: new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    };
  }

  // Generate EV battery specific predictions
  private generatePredictions(data: BatteryData): SystemPredictions {
    const batteryTemp = data.maxTemperature || 40;
    const voltage = data.batteryVoltage || 48;
    const soc = data.stateOfCharge || 80;
    const maxCellVoltage = data.maxCellVoltage || 4.0;
    const minCellVoltage = data.minCellVoltage || 3.5;
    const tempDifference = data.maxTemperature - data.minTemperature;

    return {
      batteryRisk: this.calculateBatteryThermalRisk(batteryTemp, tempDifference),
      voltageRisk: this.calculateVoltageStabilityRisk(voltage, maxCellVoltage, minCellVoltage),
      cellDegradationRisk: this.calculateCellDegradationRisk(maxCellVoltage, minCellVoltage, soc),
      overallHealth: this.calculateOverallBatteryHealth(data),
      temperatureEfficiency: this.calculateTemperatureEfficiency(batteryTemp, tempDifference),
      chargeCycles: this.estimateChargeCycles(soc, data.chargeState),
      lastUpdated: new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    };
  }

  private calculateBatteryThermalRisk(maxTemp: number, tempDifference: number): PredictionRisk {
    // Higher risk if temperature is above 45°C or if there's high temperature difference between cells
    const tempRisk = maxTemp > 45 ? (maxTemp - 45) * 8 : 0;
    const differenceRisk = tempDifference > 10 ? (tempDifference - 10) * 5 : 0;
    const riskPercentage = Math.min(100, tempRisk + differenceRisk);
    
    if (riskPercentage < 30) {
      return {
        level: 'low',
        percentage: riskPercentage,
        description: 'Battery thermal management system operating normally. Temperature distribution is even across cells.',
        recommendation: 'Continue normal operation. Regular thermal monitoring recommended.'
      };
    } else if (riskPercentage < 70) {
      return {
        level: 'medium',
        percentage: riskPercentage,
        description: 'Moderate thermal stress detected in battery pack. Temperature variations may indicate cooling system inefficiency.',
        recommendation: 'Monitor thermal patterns closely. Consider reducing charging rate if temperatures persist.'
      };
    } else {
      return {
        level: 'high',
        percentage: riskPercentage,
        description: 'Critical thermal conditions detected. High risk of thermal runaway or permanent battery damage.',
        recommendation: 'IMMEDIATE ACTION: Stop charging, reduce load, activate enhanced cooling protocol.'
      };
    }
  }

  private calculateVoltageStabilityRisk(batteryVoltage: number, maxCell: number, minCell: number): PredictionRisk {
    const optimalVoltage = 48;
    const voltageDeviation = Math.abs(batteryVoltage - optimalVoltage);
    const cellImbalance = maxCell - minCell;
    
    const voltageRisk = voltageDeviation > 5 ? voltageDeviation * 3 : 0;
    const imbalanceRisk = cellImbalance > 0.5 ? cellImbalance * 20 : 0;
    const riskPercentage = Math.min(100, voltageRisk + imbalanceRisk);

    if (riskPercentage < 30) {
      return {
        level: 'low',
        percentage: riskPercentage,
        description: 'Voltage levels stable across all battery cells. System operating within optimal parameters.',
        recommendation: 'Continue normal operation. Regular cell balancing scheduled.'
      };
    } else if (riskPercentage < 70) {
      return {
        level: 'medium',
        percentage: riskPercentage,
        description: 'Cell voltage imbalance detected. Some cells may be degrading faster than others.',
        recommendation: 'Perform cell balancing cycle. Monitor individual cell performance.'
      };
    } else {
      return {
        level: 'high',
        percentage: riskPercentage,
        description: 'Significant voltage instability or cell imbalance. Risk of system shutdown or cell damage.',
        recommendation: 'IMMEDIATE ACTION: Initiate emergency cell balancing protocol. Inspect battery connections.'
      };
    }
  }

  private calculateCellDegradationRisk(maxCell: number, minCell: number, soc: number): PredictionRisk {
    const cellImbalance = maxCell - minCell;
    const lowSOCRisk = soc < 20 ? (20 - soc) * 2 : 0;
    const highSOCRisk = soc > 90 ? (soc - 90) * 3 : 0;
    const imbalanceRisk = cellImbalance > 0.3 ? cellImbalance * 30 : 0;
    
    const riskPercentage = Math.min(100, lowSOCRisk + highSOCRisk + imbalanceRisk);

    if (riskPercentage < 30) {
      return {
        level: 'low',
        percentage: riskPercentage,
        description: 'Battery cells showing minimal signs of degradation. Capacity retention within expected parameters.',
        recommendation: 'Maintain current charging patterns. Regular capacity testing scheduled.'
      };
    } else if (riskPercentage < 70) {
      return {
        level: 'medium',
        percentage: riskPercentage,
        description: 'Moderate cell degradation detected. Some capacity loss or increased internal resistance observed.',
        recommendation: 'Adjust charging strategy to extend battery life. Avoid extreme SOC levels.'
      };
    } else {
      return {
        level: 'high',
        percentage: riskPercentage,
        description: 'Significant cell degradation detected. Battery capacity and performance severely compromised.',
        recommendation: 'IMMEDIATE ACTION: Schedule battery replacement. Limit operation to essential functions only.'
      };
    }
  }

  private calculateOverallBatteryHealth(data: BatteryData): number {
    // Simple health calculation based on multiple factors
    let health = 100;
    
    // Temperature factor
    if (data.maxTemperature > 50) health -= 15;
    else if (data.maxTemperature > 45) health -= 8;
    
    // Cell balance factor
    const cellImbalance = data.maxCellVoltage - data.minCellVoltage;
    if (cellImbalance > 0.5) health -= 20;
    else if (cellImbalance > 0.3) health -= 10;
    
    // SOC factor
    if (data.stateOfCharge < 10 || data.stateOfCharge > 95) health -= 5;
    
    // Voltage factor
    if (data.batteryVoltage < 42 || data.batteryVoltage > 54) health -= 10;
    
    return Math.max(0, Math.min(100, health));
  }

  private calculateTemperatureEfficiency(maxTemp: number, tempDifference: number): number {
    let efficiency = 100;
    
    // Optimal temperature range is 20-40°C
    if (maxTemp > 40) {
      efficiency -= (maxTemp - 40) * 2;
    } else if (maxTemp < 20) {
      efficiency -= (20 - maxTemp) * 1.5;
    }
    
    // Temperature uniformity factor
    efficiency -= tempDifference * 2;
    
    return Math.max(0, Math.min(100, efficiency));
  }

  private estimateChargeCycles(soc: number, chargeState: string): number {
    // This would normally come from persistent storage or API
    // For demo purposes, return a simulated value based on SOC patterns
    const baseValue = 150;
    const variation = Math.floor(Math.random() * 50);
    return baseValue + variation;
  }

  // Start real-time updates every 3 seconds
  private startRealTimeUpdates(): void {
    interval(3000).pipe(
      switchMap(() => this.getLatestBatteryData())
    ).subscribe({
      next: (data) => {
        if (data) {
          const metrics = this.transformToDashboardMetrics(data);
          const predictions = this.generatePredictions(data);
          
          this.metricsSubject.next(metrics);
          this.predictionsSubject.next(predictions);
        }
      },
      error: (error) => {
        console.error('Error fetching real-time data:', error);
        // Fallback to simulated data if API is unavailable
        this.generateSimulatedData();
      }
    });
  }

  // Fallback simulation when API is unavailable
  private generateSimulatedData(): void {
    const simulatedData: BatteryData = {
      id: 'sim-' + Date.now(),
      evdata: 'simulated',
      deviceId: 'EV-DEV-001',
      sensorId: 'BATT-SENSOR-001',
      type: 'battery',
      vehicleState: Math.random() > 0.5 ? 'running' : 'idle',
      chargeState: Math.random() > 0.3 ? 'discharging' : 'charging',
      batteryVoltage: this.generateRandomValue(46, 52),
      motorCurrent: this.generateRandomValue(12, 25),
      stateOfCharge: this.generateRandomValue(20, 95),
      maxCellVoltage: this.generateRandomValue(3.8, 4.2),
      minCellVoltage: this.generateRandomValue(3.5, 3.9),
      maxTemperature: this.generateRandomValue(35, 50),
      minTemperature: this.generateRandomValue(25, 35),
      timestamp: new Date().toISOString()
    };

    const metrics = this.transformToDashboardMetrics(simulatedData);
    const predictions = this.generatePredictions(simulatedData);
    
    this.metricsSubject.next(metrics);
    this.predictionsSubject.next(predictions);
  }

  private generateRandomValue(min: number, max: number): number {
    return Number((min + Math.random() * (max - min)).toFixed(2));
  }

  private handleError<T>(operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {
      console.error(`${operation} failed: ${error.message}`);
      return new Observable<T>(observer => {
        if (result !== undefined) {
          observer.next(result as T);
        }
        observer.complete();
      });
    };
  }
}