// src/app/app.component.ts
import { Component, OnInit } from '@angular/core';
import { ThemeService } from './services/theme.service';

@Component({
  selector: 'app-root',
  template: `
    <router-outlet></router-outlet>
  `,
  styles: []
})
export class AppComponent implements OnInit {
  title = 'ACTIA EV Intelligence';

  constructor(private themeService: ThemeService) {}

  ngOnInit(): void {
    // Theme service will initialize automatically
  }
}