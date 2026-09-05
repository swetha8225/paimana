import { useEffect, useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, monthLabel } from '../api'
import { ErrorBox, Loader } from '../components/ui'

export default function Trends() {
  const [d, setD] = useState<any>(null)
  const [a, setA] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    Promise.all([api('/api/forecast'), api('/api/anomalies')])
      .then(([f, an]) => { setD(f); setA(an) })
      .catch(e => setErr(e.message))
  }, [])
  if (err) return <div className="page"><ErrorBox msg={err} /></div>
  if (!d) return <Loader />
  const data = (d.monthly || []).map((r: any) => ({
    ...r, month: monthLabel(r.report_month),
    avg_cor: r.avg_cor == null ? null : +r.avg_cor.toFixed(1),
    share_over: r.share_over == null ? null : +(100 * r.share_over).toFixed(1),
  }))

  return <div className="page">
    <div className="card mb">
      <h3>Observed panel trends</h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
          <XAxis dataKey="month" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="l" tick={{ fontSize: 11 }} label={{ value: '%', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="l" type="monotone" dataKey="avg_cor" name="Avg cost overrun %" stroke="#c62828" strokeWidth={2} dot={{ r: 3 }} connectNulls />
          <Line yAxisId="l" type="monotone" dataKey="share_over" name="% projects with overrun" stroke="#ff9933" strokeWidth={2} dot={{ r: 3 }} connectNulls />
          <Line yAxisId="r" type="monotone" dataKey="exp_cr" name="Expenditure (₹ Cr)" stroke="#138808" strokeWidth={2} dot={{ r: 3 }} connectNulls />
        </LineChart>
      </ResponsiveContainer>
      <div className="note">{d.disclaimer}</div>
    </div>

    <div className="grid g2 mb">
      <div className="card">
        <h3>Per-project forecasting feasibility</h3>
        <div className="note" style={{ fontSize: 12.5 }}>{d.feasibility?.note}</div>
        <table className="tbl mt">
          <tbody>
            <tr><td>Projects in panel</td><td className="num">{d.feasibility?.projects?.toLocaleString()}</td></tr>
            <tr><td>Max monthly observations per project</td>
              <td className="num">{d.feasibility?.max_observations_per_project} (min {d.feasibility?.min_required} needed for ARIMA/ETS)</td></tr>
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3>Global panel model — hold-out evaluation</h3>
        {d.evaluation?.available ? <table className="tbl">
          <tbody>
            <tr><td>Hold-out month</td><td className="num">{monthLabel(d.evaluation.test_month)}</td></tr>
            <tr><td>Observations</td><td className="num">{d.evaluation.n_test?.toLocaleString()}</td></tr>
            <tr><td>Panel model MAE</td><td className="num">{d.evaluation.model_mae?.toFixed(2)} pp</td></tr>
            <tr><td>Naive baseline MAE</td><td className="num">{d.evaluation.naive_mae?.toFixed(2)} pp</td></tr>
            <tr><td>Verdict</td>
              <td className="num" style={{ fontWeight: 700, color: d.evaluation.model_beats_naive ? 'var(--low)' : 'var(--critical)' }}>
                {d.evaluation.model_beats_naive ? 'Model used' : 'Baseline preferred — model forecast not shown'}</td></tr>
          </tbody>
        </table> : <div className="insufficient">{d.evaluation?.reason}</div>}
      </div>
    </div>

    <div className="card">
      <h3>Statistical anomalies (latest month)</h3>
      {a?.count ? <>
        <div className="note mb">{a.note}</div>
        <table className="tbl">
          <thead><tr><th>Project</th><th>Sector</th><th className="num">Original → Current</th>
            <th className="num">COR %</th><th className="num">Slip (mo)</th></tr></thead>
          <tbody>
            {a.items.slice(0, 15).map((p: any) => <tr key={p.project_code}>
              <td className="name" title={p.project_name}>{p.project_name}</td>
              <td style={{ fontSize: 11.5 }}>{p.sector}</td>
              <td className="num" style={{ fontSize: 11.5 }}>{p.original_cost?.toFixed(0)} → {p.latest_cost?.toFixed(0)}</td>
              <td className="num">{p.cost_overrun_pct == null ? '—' : p.cost_overrun_pct.toFixed(0)}</td>
              <td className="num">{p.time_overrun_months == null ? '—' : p.time_overrun_months.toFixed(0)}</td>
            </tr>)}
          </tbody>
        </table>
      </> : <div className="insufficient">No anomalies flagged.</div>}
    </div>
  </div>
}
