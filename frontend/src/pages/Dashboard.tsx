import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, fmtCr, fmtCrFull, fmtNum, fmtPct, monthLabel, RISK_COLORS } from '../api'
import { Badge, ErrorBox, Loader } from '../components/ui'

export default function Dashboard() {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState('')
  const nav = useNavigate()
  useEffect(() => { api('/api/dashboard').then(setD).catch(e => setErr(e.message)) }, [])
  if (err) return <ErrorBox msg={err} />
  if (!d) return <Loader />
  const k = d.kpis
  const risk = Object.entries(d.risk_distribution || {}).map(([name, value]) =>
    ({ name, value: value as number }))

  return <div className="page">
    <div className="grid g4">
      <div className="card kpi accent">
        <div className="label">Ongoing projects ({monthLabel(d.latest_month)})</div>
        <div className="value">{fmtNum(k.ongoing_projects)}</div>
        <div className="sub">across {d.sectors?.length} sectors</div>
      </div>
      <div className="card kpi">
        <div className="label">Original cost</div>
        <div className="value">{fmtCr(k.original_cost_cr)}</div>
        <div className="sub">current estimate {fmtCr(k.latest_cost_cr)}</div>
      </div>
      <div className="card kpi">
        <div className="label">Cumulative expenditure</div>
        <div className="value">{fmtCr(k.expenditure_cr)}</div>
        <div className="sub">{fmtPct(100 * k.expenditure_cr / k.latest_cost_cr, 1)} of current estimate</div>
      </div>
      <div className="card kpi">
        <div className="label">Projects with cost overrun</div>
        <div className="value" style={{ color: 'var(--critical)' }}>{fmtNum(k.cost_overrun_projects)}</div>
        <div className="sub">{fmtPct(k.cost_overrun_share_pct)} of ongoing · avg {fmtPct(k.avg_cost_overrun_pct)} overrun</div>
      </div>
    </div>

    <div className="grid g4 mt">
      <div className="card kpi">
        <div className="label">Schedule overrun (anticipated)</div>
        <div className="value">{fmtNum(k.schedule_overrun_projects)}</div>
        <div className="sub">avg slip {k.avg_schedule_overrun_months?.toFixed(0) ?? '—'} months</div>
      </div>
      <div className="card kpi">
        <div className="label">High / Critical risk</div>
        <div className="value">{fmtNum((d.risk_distribution?.High || 0) + (d.risk_distribution?.Critical || 0))}</div>
        <div className="sub">by PAIMANA implementation-risk score</div>
      </div>
      <div className="card kpi">
        <div className="label">Statistical anomalies</div>
        <div className="value">{fmtNum(d.warning_counts?.find((w: any) => w.warning_type === 'anomaly')?.n || 0)}</div>
        <div className="sub">isolation-forest flagged for verification</div>
      </div>
      <div className="card kpi">
        <div className="label">Active warnings issued</div>
        <div className="value">{fmtNum(d.warning_counts?.reduce((s: number, w: any) => s + w.n, 0))}</div>
        <div className="sub">rule-based, transparent triggers</div>
      </div>
    </div>

    <div className="grid g2 mt">
      <div className="card">
        <h3>Monthly trend — ongoing projects & spend</h3>
        <ResponsiveContainer width="100%" height={230}>
          <LineChart data={(d.trend || []).map((t: any) => ({
            ...t, month: monthLabel(t.report_month),
            orig: t.orig_cr / 100000, latest: t.latest_cr / 100000, exp: t.exp_cr / 100000,
          }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: '₹ L Cr', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip formatter={(v: any, n: any) => [`${(+v).toFixed(2)} L Cr`, n]} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="orig" name="Original" stroke="#0b2b5b" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="latest" name="Current est." stroke="#ff9933" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="exp" name="Expenditure" stroke="#138808" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
        <div className="note">Report months available in the public MoSPI extracts (Apr–Sep 2024, Mar 2025).</div>
      </div>
      <div className="card">
        <h3>Implementation-risk distribution</h3>
        <ResponsiveContainer width="100%" height={230}>
          <PieChart>
            <Pie data={risk} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
              {risk.map(r => <Cell key={r.name} fill={RISK_COLORS[r.name] || '#999'} />)}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="note">Transparent component score (cost / schedule / expenditure / reporting) — see Risk Methodology. Not an official government rating.</div>
      </div>
    </div>

    <div className="grid g2 mt">
      <div className="card">
        <h3>Highest-risk projects</h3>
        <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>Project</th><th className="num">Risk</th><th className="num">COR %</th><th className="num">Delay (mo)</th></tr></thead>
          <tbody>
            {(d.top_risk || []).map((p: any) => <tr key={p.project_code}>
              <td className="name" title={p.project_name}>{p.project_name}</td>
              <td className="num"><Badge level={p.risk_level} /> {p.risk_total?.toFixed(0)}</td>
              <td className="num" style={{ color: p.cost_overrun_pct > 0 ? 'var(--critical)' : undefined }}>
                {p.cost_overrun_pct == null ? '—' : p.cost_overrun_pct.toFixed(0)}%</td>
              <td className="num">{p.time_overrun_months == null ? '—' : p.time_overrun_months.toFixed(0)}</td>
            </tr>)}
          </tbody>
        </table>
        </div>
        <div style={{ marginTop: 8 }}>
          <button className="link-btn" onClick={() => nav('/projects')}>View all projects →</button>
        </div>
      </div>
      <div className="card">
        <h3>Sector snapshot (latest month)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={d.sectors?.slice(0, 10).map((s: any) => ({
            sector: s.sector.length > 16 ? s.sector.slice(0, 15) + '…' : s.sector,
            projects: s.n, overrun: s.over,
          }))} layout="vertical" margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="sector" width={120} tick={{ fontSize: 10.5 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="projects" name="Ongoing" fill="#0b2b5b" radius={[0, 3, 3, 0]} />
            <Bar dataKey="overrun" name="With cost overrun" fill="#ff9933" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  </div>
}
