import React, { useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, Tooltip } from 'react-leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import 'leaflet/dist/leaflet.css';

// Mock GeoJSON Data for the 1km x 1km vector grid
const mockGeoJSON = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: { grid_id: "G_00000", pm25: 145, pred_24h: 155, primary_source: "Traffic" },
      geometry: { type: "Polygon", coordinates: [[[77.20, 28.60], [77.21, 28.60], [77.21, 28.61], [77.20, 28.61], [77.20, 28.60]]] }
    },
    {
      type: "Feature",
      properties: { grid_id: "G_00001", pm25: 65, pred_24h: 60, primary_source: "Construction" },
      geometry: { type: "Polygon", coordinates: [[[77.21, 28.60], [77.22, 28.60], [77.22, 28.61], [77.21, 28.61], [77.21, 28.60]]] }
    },
    {
      type: "Feature",
      properties: { grid_id: "G_00002", pm25: 35, pred_24h: 40, primary_source: "Industrial" },
      geometry: { type: "Polygon", coordinates: [[[77.20, 28.61], [77.21, 28.61], [77.21, 28.62], [77.20, 28.62], [77.20, 28.61]]] }
    }
  ]
};

// Mock Enforcement Targets
const mockEnforcementTargets = [
  { id: 'G_00000', rank: 1, coords: '28.605, 77.205', source: 'Traffic Congestion (85%)' },
  { id: 'G_00004', rank: 2, coords: '28.615, 77.225', source: 'Thermal Industrial Anomalies (92%)' }
];

// Mock Chart Data for Validation Panel
const mockChartData = [
  { time: 'T-24h', actual: 45, baseline: 45, ai_pred: 45 },
  { time: 'T-12h', actual: 50, baseline: 45, ai_pred: 52 },
  { time: 'T-0h', actual: 80, baseline: 50, ai_pred: 78 },
  { time: 'T+12h', actual: null, baseline: 80, ai_pred: 110 },
  { time: 'T+24h', actual: null, baseline: 80, ai_pred: 145 },
];

export default function AirQualityDashboard() {
  const [geoData, setGeoData] = useState(mockGeoJSON);
  const [enforcementLog, setEnforcementLog] = useState(mockEnforcementTargets);
  const [chartData, setChartData] = useState(mockChartData);

  // Dynamic polygon color based on AQI severity
  const getStyle = (feature) => {
    const pm25 = feature.properties.pm25;
    let color = '#22c55e'; // Green (Good)
    if (pm25 > 100) color = '#ef4444'; // Red (Hazardous)
    else if (pm25 > 50) color = '#f97316'; // Orange (Poor)

    return {
      fillColor: color,
      weight: 1,
      opacity: 0.6,
      color: '#ffffff',
      fillOpacity: 0.5
    };
  };

  const handleDeploy = (id) => {
    alert(`Inspector teams mobilized for target grid: ${id}`);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6 flex flex-col font-sans">

      {/* Header */}
      <header className="mb-6 flex justify-between items-center border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-blue-500">
            Smart City Air Intelligence Command Center
          </h1>
          <p className="text-sm text-gray-400 mt-1">Real-time telemetry, predictive forecasting, and localized source attribution.</p>
        </div>
        <div className="flex gap-4">
          <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-semibold border border-green-500/30">System Status: Optimal</span>
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm font-semibold border border-blue-500/30">Active Grids: 1,024</span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">

        {/* Left Column: Map Viewport (Takes up 2 cols on lg) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-gray-800 rounded-xl shadow-lg border border-gray-700 overflow-hidden h-[500px] flex flex-col">
            <div className="p-4 border-b border-gray-700 bg-gray-800/50 flex justify-between items-center">
              <h2 className="text-lg font-semibold text-gray-200">Spatio-Temporal Grid Overlay (1km x 1km)</h2>
              <select className="bg-gray-900 border border-gray-600 text-sm rounded px-2 py-1 text-gray-300 focus:outline-none">
                <option>Layer: Current PM2.5</option>
                <option>Layer: 24h Prediction</option>
                <option>Layer: Source Attribution</option>
              </select>
            </div>
            <div className="flex-1 relative z-0">
              <MapContainer center={[28.605, 77.205]} zoom={14} className="h-full w-full bg-gray-900">
                <TileLayer
                  url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                  attribution='&copy; <a href="https://carto.com/">CARTO</a>'
                />
                <GeoJSON
                  data={geoData}
                  style={getStyle}
                >
                  <Tooltip sticky className="bg-gray-800 text-gray-100 border-none shadow-xl rounded-lg p-2">
                    <div className="flex flex-col gap-1">
                      <span className="font-bold text-teal-400">Grid ID: G_XXXX</span>
                      <span className="text-sm">PM2.5: <span className="font-semibold text-white">Live Data</span></span>
                      <span className="text-sm text-yellow-300">Dominant Source</span>
                    </div>
                  </Tooltip>
                </GeoJSON>
              </MapContainer>
            </div>
          </div>

          {/* Analytics & Validation Panel */}
          <div className="bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-5 h-72">
            <h2 className="text-lg font-semibold text-gray-200 mb-4">AI Predictive Validation vs Baseline</h2>
            <ResponsiveContainer width="100%" height="80%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <RechartsTooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', color: '#F3F4F6' }} />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={3} name="Actual PM2.5" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="ai_pred" stroke="#3B82F6" strokeWidth={3} strokeDasharray="5 5" name="AI Prediction" />
                <Line type="monotone" dataKey="baseline" stroke="#EF4444" strokeWidth={2} strokeDasharray="3 3" name="Persistence Baseline" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column: Enforcement Console */}
        <div className="bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-5 flex flex-col">
          <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            Enforcement Intelligence Console
          </h2>
          <p className="text-xs text-gray-400 mb-4 border-b border-gray-700 pb-4">
            Live prioritized dispatch log optimizing for severity and source enforceability.
          </p>

          <div className="flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar">
            {enforcementLog.map((target) => (
              <div key={target.id} className="bg-gray-700/50 rounded-lg p-4 border border-gray-600 hover:border-blue-500/50 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <span className="bg-red-500/20 text-red-400 text-xs font-bold px-2 py-1 rounded">Rank #{target.rank}</span>
                  <span className="text-xs text-gray-400 font-mono">{target.id}</span>
                </div>
                <div className="mb-3">
                  <p className="text-sm font-semibold text-gray-200">GPS: <span className="font-mono text-gray-300">{target.coords}</span></p>
                  <p className="text-sm text-gray-400 mt-1">Source: <span className="text-yellow-400/90">{target.source}</span></p>
                </div>
                <button
                  onClick={() => handleDeploy(target.id)}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 rounded shadow transition-all active:scale-95"
                >
                  Deploy Inspector Teams
                </button>
              </div>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
}
