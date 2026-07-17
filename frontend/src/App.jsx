import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const App = () => {
  const [forecasts, setForecasts] = useState([]);
  
  useEffect(() => {
    // Fetch from FastAPI backend
    fetch('http://localhost:8000/api/v1/forecast/grid')
      .then(res => res.json())
      .then(data => setForecasts(data))
      .catch(err => console.error("Failed to fetch forecasts", err));
  }, []);

  const getMarkerColor = (pm25) => {
    if (pm25 > 150) return 'red';
    if (pm25 > 100) return 'orange';
    if (pm25 > 50) return 'yellow';
    return 'green';
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white font-sans">
      <header className="p-4 bg-gray-800 shadow-md">
        <h1 className="text-2xl font-bold text-blue-400">ET AI 2.0: Smart City Air Intelligence</h1>
        <p className="text-sm text-gray-400">1km x 1km Hyperlocal Forecasting & Causal Attribution</p>
      </header>
      
      <main className="flex-1 relative z-0">
        <MapContainer center={[28.6139, 77.2090]} zoom={12} className="h-full w-full">
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          />
          
          {forecasts.map((f, i) => {
            // Mocking coordinates for the grid cells
            const lat = 28.6139 + (Math.random() - 0.5) * 0.2;
            const lng = 77.2090 + (Math.random() - 0.5) * 0.2;
            
            return (
              <CircleMarker 
                key={f.grid_id} 
                center={[lat, lng]} 
                radius={8}
                pathOptions={{ 
                  color: getMarkerColor(f.forecasted_pm25_1hr), 
                  fillColor: getMarkerColor(f.forecasted_pm25_1hr),
                  fillOpacity: 0.7 
                }}
              >
                <Popup className="text-gray-900">
                  <div className="font-bold border-b pb-1 mb-1">{f.grid_id}</div>
                  <div>Current PM2.5: {f.current_pm25.toFixed(1)}</div>
                  <div className="font-bold text-red-600">1-Hr Forecast: {f.forecasted_pm25_1hr.toFixed(1)}</div>
                  <div className="text-xs text-gray-500 mt-1">Severity: {f.hotspot_severity}</div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </main>
    </div>
  );
};

export default App;
