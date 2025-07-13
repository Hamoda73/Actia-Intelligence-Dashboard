
        // Theme Toggle Functionality
        const themeToggle = document.getElementById('theme-toggle');
        const body = document.body;
        
        // Check for saved theme preference or use preferred color scheme
        const savedTheme = localStorage.getItem('theme') || 
                          (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
        
        // Apply the saved theme
        if (savedTheme === 'light') {
            body.setAttribute('data-theme', 'light');
            themeToggle.checked = true;
        }
        
        // Theme toggle event
        themeToggle.addEventListener('change', function() {
            if (this.checked) {
                body.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
            } else {
                body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
            }
        });

        // Page Navigation
        function showPage(pageId) {
            document.getElementById('metrics-page').classList.add('hidden');
            document.getElementById('predictions-page').classList.add('hidden');
            
            const page = document.getElementById(pageId + '-page');
            page.classList.remove('hidden');
            page.classList.add('fade-in');
            
            setTimeout(() => {
                page.classList.remove('fade-in');
            }, 400);
            
            // Update active nav item
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            event.currentTarget.classList.add('active');
        }
        
        // Data Simulation
        function simulateData() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            
            // Update metrics
            document.getElementById('update-time').textContent = timeString;
            document.getElementById('prediction-time').textContent = timeString;
            
            // Generate random values
            const data = {
                temperature: (35 + Math.random() * 15).toFixed(1),
                voltage: (48 + Math.random() * 4 - 2).toFixed(2),
                coolant_level: (75 + Math.random() * 25).toFixed(0),
                current: (12 + Math.random() * 8).toFixed(1),
                vibration: (0.08 + Math.random() * 0.25).toFixed(3),
                pressure: (140 + Math.random() * 40).toFixed(0),
                predictions: {
                    coolant: Math.floor(Math.random() * 20),
                    transformer: Math.floor(Math.random() * 40 + 15),
                    battery: Math.floor(Math.random() * 50 + 30)
                }
            };
            
            // Update metrics display
            document.getElementById('battery-temp').textContent = data.temperature;
            document.getElementById('voltage').textContent = data.voltage;
            document.getElementById('coolant').textContent = data.coolant_level;
            document.getElementById('current').textContent = data.current;
            document.getElementById('vibration').textContent = data.vibration;
            document.getElementById('pressure').textContent = data.pressure;
            
            // Update predictions
            updatePrediction('coolant-risk', data.predictions.coolant, 'Low', 'Medium', 'High');
            updatePrediction('transformer-risk', data.predictions.transformer, 'Low', 'Medium', 'High');
            updatePrediction('battery-risk', data.predictions.battery, 'Low', 'Medium', 'High');
        }
        
        function updatePrediction(elementId, value, lowText, mediumText, highText) {
            const element = document.getElementById(elementId);
            
            if (value < 30) {
                element.className = 'prediction-risk risk-low';
                element.textContent = `${lowText} Risk (${value}%)`;
            } else if (value < 70) {
                element.className = 'prediction-risk risk-medium';
                element.textContent = `${mediumText} Risk (${value}%)`;
            } else {
                element.className = 'prediction-risk risk-high';
                element.textContent = `${highText} Risk (${value}%)`;
            }
        }
        
        // Simulate data every 3 seconds
        setInterval(simulateData, 3000);
        simulateData(); // Initial call
    