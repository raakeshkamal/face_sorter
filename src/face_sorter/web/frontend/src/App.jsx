import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import './App.css';
import Dashboard from './views/Dashboard.jsx';
import Training from './views/Training.jsx';
import Cleaning from './views/Cleaning.jsx';
import Sorting from './views/Sorting.jsx';
import Classes from './views/Classes.jsx';
import Unsorted from './views/Unsorted.jsx';
import Toast from './components/Toast.jsx';

function App() {
  return (
    <Router>
      <div id="app">
        <div className="app-container">
          {/* Sidebar Navigation */}
          <nav className="sidebar">
            <div className="sidebar-header">
              <h1 className="app-title">Face Sorter</h1>
            </div>

            <div className="sidebar-nav">
              <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end>
                <span className="nav-icon">📊</span>
                <span>Dashboard</span>
              </NavLink>
              <NavLink to="/training" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">🎯</span>
                <span>Training</span>
              </NavLink>
              <NavLink to="/cleaning" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">🧹</span>
                <span>Cleaning</span>
              </NavLink>
              <NavLink to="/sorting" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">🔀</span>
                <span>Sorting</span>
              </NavLink>
              <NavLink to="/classes" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">👥</span>
                <span>Classes</span>
              </NavLink>
              <NavLink to="/unsorted" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
                <span className="nav-icon">📷</span>
                <span>Unsorted</span>
              </NavLink>
            </div>
          </nav>

          {/* Main Content */}
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/training" element={<Training />} />
              <Route path="/cleaning" element={<Cleaning />} />
              <Route path="/sorting" element={<Sorting />} />
              <Route path="/classes" element={<Classes />} />
              <Route path="/unsorted" element={<Unsorted />} />
            </Routes>
          </main>
        </div>

        {/* Toast Notifications */}
        <Toast />
      </div>
    </Router>
  );
}

export default App;