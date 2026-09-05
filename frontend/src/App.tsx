import { NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { BrowserRouter } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import ProjectDetail from './pages/ProjectDetail'
import ModelPerformance from './pages/ModelPerformance'
import DataQuality from './pages/DataQuality'
import Trends from './pages/Trends'
import Assistant from './pages/Assistant'
import RiskMethodology from './pages/RiskMethodology'
import { api, monthLabel } from './api'
import { useEffect, useState } from 'react'

const NAV = [
  { to: '/', icon: '📊', label: 'Dashboard', end: true },
  { to: '/projects', icon: '🏗️', label: 'Projects' },
  { to: '/risk-methodology', icon: '⚖️', label: 'Risk Methodology' },
  { to: '/model-performance', icon: '🤖', label: 'Model Performance' },
  { to: '/trends', icon: '📈', label: 'Trends & Forecast' },
  { to: '/data-quality', icon: '🗂️', label: 'Data & Sources' },
  { to: '/assistant', icon: '💬', label: 'Assistant' },
]

function Shell() {
  const loc = useLocation()
  const [meta, setMeta] = useState<any>(null)
  const [navOpen, setNavOpen] = useState(false)
  useEffect(() => {
    api('/api/meta').then(setMeta).catch(() => setMeta(null))
  }, [])
  // close the mobile menu on navigation + set the browser tab title
  useEffect(() => {
    setNavOpen(false)
    const current = NAV.find(n =>
      n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to))
    document.title = `${current ? current.label + ' · ' : ''}PAIMANA`
  }, [loc.pathname])

  const title = NAV.find(n =>
    n.end ? loc.pathname === n.to : loc.pathname.startsWith(n.to))?.label || 'PAIMANA'

  return (
    <div className="app">
      {navOpen && <div className="backdrop" onClick={() => setNavOpen(false)} />}
      <aside className={`sidebar ${navOpen ? 'open' : ''}`}>
        <button className="sidebar-close" onClick={() => setNavOpen(false)} aria-label="Close menu">✕</button>
        <div className="brand">
          <h1>PAIMANA</h1>
          <div className="sub">Pro-active Analytics for Infrastructure Monitoring &amp; Assessment (National Analytics)</div>
          <div className="dev">SMART INDIA HACKATHON 2026 · SIH26103</div>
        </div>
        <nav className="nav">
          {NAV.map(n => (
            <NavLink key={n.to} to={n.to} end={n.end as any}
              className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="icon">{n.icon}</span><span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="foot">
          Built on public MoSPI QPISR &amp; Flash Report data.
          {meta ? <><br />Latest report: {monthLabel(meta.dataset?.report_months?.slice(-1)[0])}
            <br />{meta.dataset?.unique_projects?.toLocaleString()} projects · {meta.dataset?.report_months?.length} months</> : null}
        </div>
      </aside>
      <div className="main">
        <div className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button className="menu-btn" onClick={() => setNavOpen(true)} aria-label="Open menu">☰</button>
            <h2>{title}</h2>
          </div>
          <div className="meta">
            {meta ? `MoSPI Central Sector Projects · ${meta.dataset?.panel_rows?.toLocaleString()} records` : ''}
          </div>
        </div>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:code" element={<ProjectDetail />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/data-quality" element={<DataQuality />} />
          <Route path="/trends" element={<Trends />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/risk-methodology" element={<RiskMethodology />} />
        </Routes>
      </div>
    </div>
  )
}

export default function App() {
  return <BrowserRouter><Shell /></BrowserRouter>
}
