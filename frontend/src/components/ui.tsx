import React from 'react'
import { useParams, useNavigate } from 'react-router-dom'

export const Loader = () => <div className="loading">Loading…</div>
export const ErrorBox = ({ msg }: { msg: string }) =>
  <div className="error">⚠ {msg}</div>

export const Insufficient = ({ children }: { children?: React.ReactNode }) => (
  <div className="insufficient">Insufficient data available for this analysis.
    {children ? <div style={{ marginTop: 6 }}>{children}</div> : null}
  </div>
)

export const Badge = ({ level }: { level: string }) =>
  <span className={`badge ${level}`}>{level}</span>

export const SevBadge = ({ severity }: { severity: string }) =>
  <span className={`badge sev-${severity}`}>{severity}</span>

export function useProjectLink() {
  const nav = useNavigate()
  return (code: string) => nav(`/projects/${code}`)
}

export const RiskBar = ({ components }: { components: Record<string, number> }) => {
  const colors: Record<string, string> = {
    cost_risk: 'var(--critical)', schedule_risk: 'var(--high)',
    expenditure_risk: 'var(--moderate)', reporting_risk: '#8da2bd',
  }
  const labels: Record<string, string> = {
    cost_risk: 'Cost risk', schedule_risk: 'Schedule risk',
    expenditure_risk: 'Expenditure risk', reporting_risk: 'Reporting/data risk',
  }
  return <div>
    {Object.entries(components || {}).map(([k, v]) => (
      <div className="component" key={k}>
        <div className="row"><span>{labels[k] || k}</span><span>{v?.toFixed?.(1)}</span></div>
        <div className="bar"><div style={{
          width: `${Math.min(100, Math.max(0, v))}%`, background: colors[k] || '#999',
        }} /></div>
      </div>
    ))}
  </div>
}

export const fmtMoney = (v: number | null | undefined) => {
  if (v == null || Number.isNaN(v)) return '—'
  if (v >= 100000) return `₹${(v / 100000).toFixed(2)} L Cr`
  return `₹${Math.round(v).toLocaleString('en-IN')} Cr`
}
