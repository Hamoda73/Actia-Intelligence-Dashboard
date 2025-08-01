// src/app/services/theme.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private isDarkThemeSubject = new BehaviorSubject<boolean>(true);
  public isDarkTheme$ = this.isDarkThemeSubject.asObservable();

  constructor() {
    this.initializeTheme();
  }

  private initializeTheme(): void {
    // Check for saved theme preference or use system preference
    const savedTheme = this.getStoredTheme();
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const isDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    this.setTheme(isDark);
  }

  private getStoredTheme(): string | null {
    try {
      return localStorage.getItem('theme');
    } catch {
      return null; // Fallback for environments without localStorage
    }
  }

  private setStoredTheme(theme: string): void {
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // Ignore localStorage errors
    }
  }

  public setTheme(isDark: boolean): void {
    this.isDarkThemeSubject.next(isDark);
    
    if (isDark) {
      document.body.setAttribute('data-theme', 'dark');
      this.setStoredTheme('dark');
    } else {
      document.body.setAttribute('data-theme', 'light');
      this.setStoredTheme('light');
    }
  }

  public toggleTheme(): void {
    const currentTheme = this.isDarkThemeSubject.value;
    this.setTheme(!currentTheme);
  }

  public get isDarkTheme(): boolean {
    return this.isDarkThemeSubject.value;
  }
}