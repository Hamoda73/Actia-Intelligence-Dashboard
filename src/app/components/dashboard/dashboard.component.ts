// src/app/components/dashboard/dashboard.component.ts
import { Component, OnInit, OnDestroy, AfterViewInit } from '@angular/core';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { BatteryDataService } from '../../services/battery-data.service';
import { ThemeService } from '../../services/theme.service';
import { DashboardMetrics } from 'src/app/models/dashboard-metrics.model';
import { SystemPredictions } from 'src/app/models/pediction.model';

// Declare Chart for global access
declare var Chart: any;

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit, OnDestroy, AfterViewInit {
  private destroy$ = new Subject<void>();
 
  currentPage: 'metrics' | 'predictions' = 'metrics';
  metrics: DashboardMetrics | null = null;
  predictions: SystemPredictions | null = null;
  isDarkTheme: boolean = true;

  // Chart instances
  private voltageCurrentChart: any = null;
  private temperatureChart: any = null;
  private cellVoltageChart: any = null;
  private socChart: any = null;

  // Data storage for charts
  private timeLabels: string[] = [];
  private voltageData: number[] = [];
  private currentData: number[] = [];
  private maxTempData: number[] = [];
  private minTempData: number[] = [];
  private socData: number[] = [];
  private maxCellData: number[] = [];
  private minCellData: number[] = [];

  private readonly MAX_DATA_POINTS = 20;
  private chartsInitialized = false;

  constructor(
    private batteryDataService: BatteryDataService,
    private themeService: ThemeService
  ) {}

  ngOnInit(): void {
    // Check if Chart.js is loaded
    this.checkChartJSAvailability();

    // Subscribe to theme changes
    this.themeService.isDarkTheme$
      .pipe(takeUntil(this.destroy$))
      .subscribe(isDark => {
        this.isDarkTheme = isDark;
        this.updateChartThemes();
      });

    // Subscribe to metrics updates
    this.batteryDataService.metrics$
      .pipe(takeUntil(this.destroy$))
      .subscribe(metrics => {
        console.log('Received metrics:', metrics); // Debug log
        this.metrics = metrics;
        if (metrics) {
          this.updateChartData(metrics);
          // Initialize charts if they haven't been initialized yet and we're on metrics page
          if (!this.chartsInitialized && this.currentPage === 'metrics') {
            setTimeout(() => this.initializeCharts(), 100);
          }
        }
      });

    // Subscribe to predictions updates
    this.batteryDataService.predictions$
      .pipe(takeUntil(this.destroy$))
      .subscribe(predictions => {
        this.predictions = predictions;
      });
  }

  ngAfterViewInit(): void {
    // Initialize charts after view is ready with multiple attempts
    this.initializeChartsWithRetry();
  }

  ngOnDestroy(): void {
    this.destroyCharts();
    this.destroy$.next();
    this.destroy$.complete();
  }

  private checkChartJSAvailability(): void {
    if (typeof Chart === 'undefined') {
      console.error('Chart.js is not loaded. Please make sure Chart.js CDN is included in index.html');
      // Try to load Chart.js dynamically
      this.loadChartJSDynamically();
    } else {
      console.log('Chart.js is available:', Chart.version);
    }
  }

  private loadChartJSDynamically(): void {
    // Remove any existing Chart.js script to avoid conflicts
    const existingScripts = document.querySelectorAll('script[src*="chart"]');
    existingScripts.forEach(script => script.remove());

    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.js';
    script.onload = () => {
      console.log('Chart.js loaded dynamically');
      // Wait a bit longer for the global Chart object to be available
      setTimeout(() => {
        if (typeof Chart !== 'undefined') {
          console.log('Chart.js is now available:', Chart.version);
          this.initializeChartsWithRetry();
        } else {
          console.error('Chart.js loaded but global Chart object not available');
        }
      }, 1000);
    };
    script.onerror = () => {
      console.error('Failed to load Chart.js dynamically');
    };
    document.head.appendChild(script);
  }

  private initializeChartsWithRetry(attempts: number = 0): void {
    const maxAttempts = 5;
    
    if (typeof Chart === 'undefined' && attempts < maxAttempts) {
      console.log(`Chart.js not ready, attempt ${attempts + 1}/${maxAttempts}`);
      setTimeout(() => this.initializeChartsWithRetry(attempts + 1), 1000);
      return;
    }

    if (typeof Chart === 'undefined') {
      console.error('Chart.js failed to load after multiple attempts');
      return;
    }

    if (this.currentPage === 'metrics') {
      setTimeout(() => {
        this.initializeCharts();
      }, 500);
    }
  }

  showPage(pageId: 'metrics' | 'predictions', event?: Event): void {
    if (event) {
      const navItems = document.querySelectorAll('.nav-item');
      navItems.forEach(item => item.classList.remove('active'));
      (event.currentTarget as HTMLElement).classList.add('active');
    }
    
    this.currentPage = pageId;
    
    // Initialize charts when metrics page is shown
    if (pageId === 'metrics' && typeof Chart !== 'undefined') {
      setTimeout(() => {
        this.initializeCharts();
      }, 300);
    }
  }

  toggleTheme(): void {
    this.themeService.toggleTheme();
  }

  getRiskClass(level: string): string {
    return `risk-${level}`;
  }

  getRiskText(risk: any): string {
    if (!risk) return '';
    return `${this.capitalizeFirst(risk.level)} Risk (${risk.percentage}%)`;
  }

  getHealthColor(health: number): string {
    if (health >= 80) return 'var(--success)';
    if (health >= 60) return 'var(--warning)';
    return 'var(--danger)';
  }

  private capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  private initializeCharts(): void {
    if (typeof Chart === 'undefined') {
      console.error('Chart.js is not available');
      return;
    }

    if (this.chartsInitialized && this.currentPage === 'metrics') {
      console.log('Charts already initialized');
      return;
    }

    console.log('Initializing charts...');
    
    // Destroy existing charts first
    this.destroyCharts();
    
    // Wait for DOM elements to be available
    setTimeout(() => {
      const canvasElements = [
        'voltageCurrentChart',
        'temperatureChart', 
        'cellVoltageChart',
        'socChart'
      ];

      let elementsFound = 0;
      canvasElements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
          elementsFound++;
          console.log(`Found canvas element: ${id}`);
        } else {
          console.warn(`Canvas element not found: ${id}`);
        }
      });

      if (elementsFound === canvasElements.length) {
        // Initialize all charts
        this.initVoltageCurrentChart();
        this.initTemperatureChart();
        this.initCellVoltageChart();
        this.initSOCChart();
        this.chartsInitialized = true;
        console.log('All charts initialized successfully');
      } else {
        console.error(`Only ${elementsFound}/${canvasElements.length} canvas elements found`);
        // Retry after a short delay
        setTimeout(() => this.initializeCharts(), 1000);
      }
    }, 100);
  }

  private initVoltageCurrentChart(): void {
    const canvas = document.getElementById('voltageCurrentChart') as HTMLCanvasElement;
    if (!canvas) {
      console.error('voltageCurrentChart canvas not found');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Could not get 2D context for voltageCurrentChart');
      return;
    }

    try {
      this.voltageCurrentChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'Battery Voltage (V)',
              data: this.voltageData,
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              borderWidth: 2,
              fill: true,
              tension: 0.4,
              yAxisID: 'y'
            },
            {
              label: 'Motor Current (A)',
              data: this.currentData,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.1)',
              borderWidth: 2,
              fill: true,
              tension: 0.4,
              yAxisID: 'y1'
            }
          ]
        },
        options: this.getChartOptions({
          y: { position: 'left', title: 'Voltage (V)' },
          y1: { position: 'right', title: 'Current (A)' }
        })
      });
      console.log('voltageCurrentChart initialized');
    } catch (error) {
      console.error('Error initializing voltageCurrentChart:', error);
    }
  }

  private initTemperatureChart(): void {
    const canvas = document.getElementById('temperatureChart') as HTMLCanvasElement;
    if (!canvas) {
      console.error('temperatureChart canvas not found');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Could not get 2D context for temperatureChart');
      return;
    }

    try {
      this.temperatureChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'Max Temperature (°C)',
              data: this.maxTempData,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              borderWidth: 2,
              fill: false,
              tension: 0.4
            },
            {
              label: 'Min Temperature (°C)',
              data: this.minTempData,
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.1)',
              borderWidth: 2,
              fill: '+1',
              tension: 0.4
            }
          ]
        },
        options: this.getChartOptions({
          y: { position: 'left', title: 'Temperature (°C)' }
        })
      });
      console.log('temperatureChart initialized');
    } catch (error) {
      console.error('Error initializing temperatureChart:', error);
    }
  }

  private initCellVoltageChart(): void {
    const canvas = document.getElementById('cellVoltageChart') as HTMLCanvasElement;
    if (!canvas) {
      console.error('cellVoltageChart canvas not found');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Could not get 2D context for cellVoltageChart');
      return;
    }

    try {
      this.cellVoltageChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'Max Cell Voltage (V)',
              data: this.maxCellData,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              borderWidth: 2,
              fill: false,
              tension: 0.4
            },
            {
              label: 'Min Cell Voltage (V)',
              data: this.minCellData,
              borderColor: '#f97316',
              backgroundColor: 'rgba(249, 115, 22, 0.1)',
              borderWidth: 2,
              fill: '+1',
              tension: 0.4
            }
          ]
        },
        options: this.getChartOptions({
          y: { position: 'left', title: 'Cell Voltage (V)' }
        })
      });
      console.log('cellVoltageChart initialized');
    } catch (error) {
      console.error('Error initializing cellVoltageChart:', error);
    }
  }

  private initSOCChart(): void {
    const canvas = document.getElementById('socChart') as HTMLCanvasElement;
    if (!canvas) {
      console.error('socChart canvas not found');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Could not get 2D context for socChart');
      return;
    }

    try {
      this.socChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: this.timeLabels,
          datasets: [
            {
              label: 'State of Charge (%)',
              data: this.socData,
              borderColor: '#8b5cf6',
              backgroundColor: 'rgba(139, 92, 246, 0.2)',
              borderWidth: 3,
              fill: true,
              tension: 0.4,
              pointBackgroundColor: '#8b5cf6',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 2,
              pointRadius: 4
            }
          ]
        },
        options: {
          ...this.getChartOptions({
            y: { position: 'left', title: 'Charge (%)' }
          }),
          scales: {
            ...this.getChartOptions({}).scales,
            y: {
              ...this.getChartOptions({}).scales?.y,
              min: 0,
              max: 100,
              title: {
                display: true,
                text: 'Charge (%)',
                color: 'rgba(255, 255, 255, 0.7)'
              }
            }
          }
        }
      });
      console.log('socChart initialized');
    } catch (error) {
      console.error('Error initializing socChart:', error);
    }
  }

  private getChartOptions(scales: any): any {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      plugins: {
        legend: {
          labels: {
            color: 'rgba(255, 255, 255, 0.8)',
            font: {
              size: 12
            }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: 'white',
          bodyColor: 'white',
          borderColor: 'rgba(255, 255, 255, 0.3)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)',
            maxTicksLimit: 8
          }
        },
        ...Object.keys(scales).reduce((acc, key) => {
          acc[key] = {
            type: 'linear',
            display: true,
            position: scales[key].position,
            grid: {
              color: key === 'y' ? 'rgba(255, 255, 255, 0.1)' : 'transparent'
            },
            ticks: {
              color: 'rgba(255, 255, 255, 0.7)'
            },
            title: {
              display: true,
              text: scales[key].title,
              color: 'rgba(255, 255, 255, 0.7)'
            }
          };
          return acc;
        }, {} as any)
      },
      animation: {
        duration: 750,
        easing: 'easeInOutQuart'
      }
    };
  }

  private updateChartData(metrics: DashboardMetrics): void {
    const currentTime = new Date().toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });

    // Add new data
    this.timeLabels.push(currentTime);
    this.voltageData.push(metrics.batteryVoltage);
    this.currentData.push(metrics.motorCurrent);
    this.maxTempData.push(metrics.maxTemperature);
    this.minTempData.push(metrics.minTemperature);
    this.socData.push(metrics.stateOfCharge);
    this.maxCellData.push(metrics.maxCellVoltage);
    this.minCellData.push(metrics.minCellVoltage);

    // Keep only last MAX_DATA_POINTS
    if (this.timeLabels.length > this.MAX_DATA_POINTS) {
      this.timeLabels.shift();
      this.voltageData.shift();
      this.currentData.shift();
      this.maxTempData.shift();
      this.minTempData.shift();
      this.socData.shift();
      this.maxCellData.shift();
      this.minCellData.shift();
    }

    // Update charts if they exist
    this.updateChart(this.voltageCurrentChart);
    this.updateChart(this.temperatureChart);
    this.updateChart(this.cellVoltageChart);
    this.updateChart(this.socChart);
  }

  private updateChart(chart: any): void {
    if (chart) {
      chart.update('none');
    }
  }

  private updateChartThemes(): void {
    // Update chart colors based on theme
    if (this.voltageCurrentChart) this.voltageCurrentChart.update();
    if (this.temperatureChart) this.temperatureChart.update();
    if (this.cellVoltageChart) this.cellVoltageChart.update();
    if (this.socChart) this.socChart.update();
  }

  private destroyCharts(): void {
    if (this.voltageCurrentChart) {
      this.voltageCurrentChart.destroy();
      this.voltageCurrentChart = null;
    }
    if (this.temperatureChart) {
      this.temperatureChart.destroy();
      this.temperatureChart = null;
    }
    if (this.cellVoltageChart) {
      this.cellVoltageChart.destroy();
      this.cellVoltageChart = null;
    }
    if (this.socChart) {
      this.socChart.destroy();
      this.socChart = null;
    }
    this.chartsInitialized = false;
  }
}