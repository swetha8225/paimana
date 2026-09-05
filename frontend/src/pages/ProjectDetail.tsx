import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from 'recharts'
import { api, fmtProb, monthLabel } from '../api'
import { Badge, ErrorBox, Loader, RiskBar, SevBadge, fmtMoney } from '../components/ui'

export default function ProjectDetail() {
  const { code } = useParams()
  const nav = useNavigate()
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState('')
  const [wi, setWi] = useState<any>(null)
  const [wiErr, setWiErr] = useState('')
  const [sim, setSim] = useState<any>({})

  useEffect(() => {
    setD(null); setWi(null)
    api(`/api/projects/${code}`).then(setD).catch(e => setErr(e.message))
  }, [code])

  const latest = d?.history?.slice(-1)[0]
  const score = d?.score

  const runSim = async () => {
    setWiErr(''); setWi(null)
    const body: any = {}
    if (sim.sector) body.sector = sim.sector
    if (sim.original_cost) body.original_cost = Number(sim.original_cost)
    if (sim.planned_duration_months) body.planned_duration_months = Number(sim.planned_duration_months)
    if (sim.cumulative_expenditure) body.cumulative_expenditure = Number(sim.cumulative_expenditure)
    try {
      setWi(await api(`/api/projects/${code}/whatif`, {
        method: 'POST', body: JSON.stringify(body),
      }))
    } catch (e: any) { setWiErr(e.message) }
  }

  if (err) return <div className="page"><ErrorBox msg={err} /></div>
  if (!d) return <Loader />
  if (!latest) return <div className="page"><ErrorBox msg="Project not found" /></div>

  const shap = d.shap?.available ? d.shap.features : []
  const shapData = shap.map((f: any) => ({ name: f.feature, value: f.shap }))

  return <div className="page">
    <button className="back-btn" onClick={() => nav('/projects')}>← Back to all projects</button>
    <div className="head">
      <div className="title">
        <h2>{latest.project_name}</h2>
        <div className="sub">
          {latest.project_code} · {latest.ministry} · {latest.sector} · {latest.state || '—'}
          {latest.agency ? ` · ${latest.agency}` : ''}
        </div>
        <div style={{ marginTop: 6 }}>
          <span className="chip">Approved {latest.approval_date || '—'}</span>
          <span className="chip">Original target {latest.original_completion_target || '—'}</span>
          <span className="chip">Current target {latest.anticipated_completion_target || latest.revised_completion_target || '—'}</span>
          {latest.event !== 'ongoing_report' &&
            <span className="chip">Status: {latest.event}</span>}
        </div>
      </div>
      {score && <div className="card risk-gauge" style={{ minWidth: 170 }}>
        <div className="label" style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase' }}>Implementation risk</div>
        <div className="score" style={{ color: `var(--${score.risk_level.toLowerCase()})` }}>{score.risk_total}</div>
        <div className="lvl"><Badge level={score.risk_level} /></div>
        <div className="note" style={{ marginTop: 6 }}>as of {monthLabel(score.report_month)}</div>
      </div>}
    </div>

    <div className="grid g4">
      <div className="card">
        <h3>Cost</h3>
        <table className="tbl">
          <tbody>
            <tr><td>Original</td><td className="num">{fmtMoney(latest.original_cost)}</td></tr>
            <tr><td>Revised</td><td className="num">{fmtMoney(latest.revised_cost)}</td></tr>
            <tr><td>Anticipated</td><td className="num">{fmtMoney(latest.anticipated_cost)}</td></tr>
            <tr><td>Expenditure</td><td className="num">{fmtMoney(latest.cumulative_expenditure)}</td></tr>
            <tr><td>Overrun</td>
              <td className="num" style={{ color: latest.cost_overrun_pct > 0 ? 'var(--critical)' : 'var(--low)', fontWeight: 700 }}>
                {latest.cost_overrun_pct == null ? '—' : `${latest.cost_overrun_pct.toFixed(1)}%`}</td></tr>
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3>Schedule</h3>
        <table className="tbl">
          <tbody>
            <tr><td>Planned duration</td><td className="num">{latest.planned_duration_months?.toFixed(0) ?? '—'} mo</td></tr>
            <tr><td>Age</td><td className="num">{latest.project_age_months?.toFixed(0) ?? '—'} mo</td></tr>
            <tr><td>Elapsed fraction</td><td className="num">{latest.elapsed_fraction == null ? '—' : `${(latest.elapsed_fraction * 100).toFixed(0)}%`}</td></tr>
            <tr><td>Anticipated slip</td>
              <td className="num" style={{ color: latest.time_overrun_months > 0 ? 'var(--high)' : undefined, fontWeight: 700 }}>
                {latest.time_overrun_months == null ? '—' : `${latest.time_overrun_months.toFixed(0)} mo`}</td></tr>
            <tr><td>Physical progress</td><td className="num">{latest.physical_progress_pct == null ? 'not reported' : `${latest.physical_progress_pct.toFixed(1)}%`}</td></tr>
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3>Model predictions</h3>
        <table className="tbl">
          <tbody>
            <tr><td>P(cost overrun) — approval stage</td>
              <td className="num">{fmtProb(score?.pred_prob_approval)}</td></tr>
            <tr><td>P(cost overrun) — monitoring stage</td>
              <td className="num">{fmtProb(score?.pred_prob_monitoring)}</td></tr>
            <tr><td>Predicted overrun magnitude</td>
              <td className="num">{score?.pred_cost_overrun_pct == null ? '—' : `${score.pred_cost_overrun_pct}%`}</td></tr>
            <tr><td>Predicted delay</td>
              <td className="num">{score?.pred_time_overrun_months == null ? '—' : `${score.pred_time_overrun_months} mo`}</td></tr>
          </tbody>
        </table>
        <div className="note" style={{ marginTop: 8 }}>
          Probabilities from calibrated gradient-boosting models trained on
          MoSPI panel data (time-aware validation). The honest generalisation
          figure for new projects is shown on the Model Performance page.
        </div>
      </div>
      <div className="card">
        <h3>Risk components</h3>
        {score ? <RiskBar components={(() => { try { return JSON.parse(score.risk_components) } catch { return {} } })()} /> :
          <div className="note">No score computed.</div>}
        <div className="note" style={{ marginTop: 8 }}>Formulas & configurable weights: Risk Methodology page.</div>
      </div>
    </div>

    <div className="grid g2 mt">
      <div className="card">
        <h3>Why the model predicts this — SHAP</h3>
        {d.shap?.available ? <>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={shapData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 10.5 }} />
              <Tooltip />
              <Bar dataKey="value">
                {shapData.map((s: any, i: number) =>
                  <Cell key={i} fill={s.value > 0 ? 'var(--critical)' : 'var(--low)'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="note">Red = increases the model's predicted overrun risk; green = decreases it. {d.shap.caveat}</div>
        </> : <div className="insufficient">Explanation not available for this project.</div>}
      </div>

      <div className="card">
        <h3>Warnings & alerts</h3>
        {d.warnings?.length ? <div className="tbl-wrap"><table className="tbl">
          <thead><tr><th>Severity</th><th>Warning</th><th>Evidence</th></tr></thead>
          <tbody>
            {d.warnings.map((w: any, i: number) => <tr key={i}>
              <td><SevBadge severity={w.severity} /></td>
              <td>{w.what}</td>
              <td style={{ color: 'var(--muted)', fontSize: 11.5 }}>{w.reason}</td>
            </tr>)}
          </tbody>
        </table></div> : <div className="note">No active warnings.</div>}
        <h3 style={{ marginTop: 14 }}>Recommendations</h3>
        {d.recommendations?.map((r: any, i: number) => <div key={i} style={{ marginBottom: 8 }}>
          <span className="tag-sim">{r.based_on}</span>
          <div style={{ fontSize: 12.5, marginTop: 3 }}>→ {r.recommendation}</div>
        </div>)}
        <div className="note" style={{ marginTop: 8 }}>Rule-based suggestions derived from active warnings. PAIMANA does not state or imply official government decisions.</div>
      </div>
    </div>

    <div className="grid g2 mt">
      <div className="card">
        <h3>Similar projects (benchmarking)</h3>
        {d.peers?.available ? <>
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Peer project</th><th className="num">Orig cost</th><th className="num">COR %</th><th className="num">Delay</th></tr></thead>
            <tbody>
              {d.peers.peers.slice(0, 8).map((p: any) => <tr key={p.project_code}>
                <td className="name" title={p.project_name}>{p.project_name}</td>
                <td className="num">{fmtMoney(p.original_cost)}</td>
                <td className="num" style={{ color: p.cost_overrun_pct > 0 ? 'var(--critical)' : undefined }}>
                  {p.cost_overrun_pct == null ? '—' : p.cost_overrun_pct.toFixed(0)}</td>
                <td className="num">{p.time_overrun_months == null ? '—' : p.time_overrun_months.toFixed(0)}</td>
              </tr>)}
            </tbody>
          </table></div>
          <div className="note" style={{ marginTop: 6 }}>{d.peers.note}</div>
        </> : <div className="note">Peer benchmarking not available for this project.</div>}
      </div>

      <div className="card">
        <h3>What-if simulation</h3>
        <div className="note mb">Change project parameters and re-run the model. Results are a <b>model simulation</b> — not a government forecast.</div>
        <div className="whatif-grid">
          <div>
            <label>Sector</label>
            <input placeholder={latest.sector} value={sim.sector || ''}
              onChange={e => setSim({ ...sim, sector: e.target.value })} />
          </div>
          <div>
            <label>Original cost (₹ Cr)</label>
            <input type="number" placeholder={latest.original_cost?.toFixed(0)} value={sim.original_cost || ''}
              onChange={e => setSim({ ...sim, original_cost: e.target.value })} />
          </div>
          <div>
            <label>Planned duration (months)</label>
            <input type="number" placeholder={latest.planned_duration_months?.toFixed(0)} value={sim.planned_duration_months || ''}
              onChange={e => setSim({ ...sim, planned_duration_months: e.target.value })} />
          </div>
          <div>
            <label>Cumulative expenditure (₹ Cr)</label>
            <input type="number" placeholder={latest.cumulative_expenditure?.toFixed(0)} value={sim.cumulative_expenditure || ''}
              onChange={e => setSim({ ...sim, cumulative_expenditure: e.target.value })} />
          </div>
        </div>
        <button onClick={runSim} style={{
          marginTop: 10, padding: '8px 18px', background: 'var(--navy)', color: '#fff',
          border: 'none', borderRadius: 8, cursor: 'pointer', fontWeight: 600,
        }}>Run simulation</button>
        {wiErr && <div className="error">⚠ {wiErr}</div>}
        {wi && <>
          <div className="tbl-wrap"><table className="tbl mt">
            <thead><tr><th></th><th className="num">Before</th><th className="num">After</th></tr></thead>
            <tbody>
              <tr><td>P(overrun) — approval stage</td>
                <td className="num">{fmtProb(wi.before.pred_prob_approval)}</td>
                <td className="num">{fmtProb(wi.after.pred_prob_approval)}</td></tr>
              <tr><td>P(overrun) — monitoring stage</td>
                <td className="num">{fmtProb(wi.before.pred_prob_monitoring)}</td>
                <td className="num">{fmtProb(wi.after.pred_prob_monitoring)}</td></tr>
              <tr><td>Predicted overrun %</td>
                <td className="num">{wi.before.pred_cost_overrun_pct ?? '—'}</td>
                <td className="num">{wi.after.pred_cost_overrun_pct ?? '—'}</td></tr>
              <tr><td>Risk score</td>
                <td className="num">{wi.before.risk.total.toFixed(1)} ({wi.before.risk.level})</td>
                <td className="num">{wi.after.risk.total.toFixed(1)} ({wi.after.risk.level})</td></tr>
            </tbody>
          </table></div>
          <div className="note" style={{ marginTop: 8 }}>{wi.note}</div>
        </>}
      </div>
    </div>

    <div className="card mt">
      <h3>Reported history (all months)</h3>
      <div className="tbl-wrap">
      <table className="tbl">
        <thead><tr>
          <th>Month</th><th className="num">Original</th><th className="num">Revised</th>
          <th className="num">Anticipated</th><th className="num">Expenditure</th>
          <th className="num">Progress %</th><th className="num">COR %</th>
          <th className="num">Slip (mo)</th><th>Source</th>
        </tr></thead>
        <tbody>
          {d.history.map((h: any) => <tr key={h.report_month}>
            <td>{monthLabel(h.report_month)}{h.event !== 'ongoing_report' ? ` (${h.event})` : ''}</td>
            <td className="num">{fmtMoney(h.original_cost)}</td>
            <td className="num">{fmtMoney(h.revised_cost)}</td>
            <td className="num">{fmtMoney(h.anticipated_cost)}</td>
            <td className="num">{fmtMoney(h.cumulative_expenditure)}</td>
            <td className="num">{h.physical_progress_pct == null ? '—' : h.physical_progress_pct.toFixed(1)}</td>
            <td className="num">{h.cost_overrun_pct == null ? '—' : h.cost_overrun_pct.toFixed(1)}</td>
            <td className="num">{h.time_overrun_months == null ? '—' : h.time_overrun_months.toFixed(0)}</td>
            <td style={{ fontSize: 10.5, color: 'var(--muted)' }}>{h.source_report}</td>
          </tr>)}
        </tbody>
      </table>
      </div>
    </div>
  </div>
}
